/* twin.c -- ロータリー充填機 仮想モデル / 物理コア（params.json rev.3）
 *
 * **py/ref.py が仕様書。** 式・符号・積分の順序をそのまま写してある。
 * 速度のために順序を入れ替えていない。入れ替えると突き合わせが成立しない。
 *
 * py/ref.py の関数との対応:
 *   modified_sine        -> ms_curve
 *   cam_input_angle      -> cam_input_angle
 *   table_motion         -> twin_table_motion
 *   step_slosh           -> step_slosh
 *   liquid_force         -> liquid_force
 *   rigid_load_inertia   -> rigid_load_inertia
 *   apply_spill          -> apply_spill
 *   vertical_bending_moment -> bending_moment
 *   simulate             -> twin_step（1 ステップぶんに分解してある）
 *
 * MODEL.md と食い違うところ（py/ref.py に合わせた側）は README.md に名指しで書いた。
 */

#include "twin.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================== */
/* 最小 JSON パーサ                                                    */
/*   params.json はネストが浅いので、必要なキーだけをフラットな          */
/*   "section.key" のパスに畳んで数値だけを保持する。                    */
/*   配列は中身を読み飛ばす（viz.cameras のような表示用の配列）。        */
/* ================================================================== */

#define JMAX_ENTRIES 320
#define JPATH_LEN    96

typedef struct {
    char   path[JPATH_LEN];
    double num;
} jentry;

typedef struct {
    jentry e[JMAX_ENTRIES];
    int    n;
    int    ok;
    int    skip;        /* >0 の間は値を保存しない（配列の中身） */
    char   err[192];
    const char *base;   /* エラー位置算出用 */
} jdoc;

static void jerr(jdoc *d, const char *p, const char *what)
{
    if (!d->ok) return;              /* 最初のエラーだけ残す */
    d->ok = 0;
    snprintf(d->err, sizeof(d->err),
             "JSON 解析エラー: %s (オフセット %ld)",
             what, (long)(p - d->base));
}

static const char *jskip_ws(const char *p)
{
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    return p;
}

/* 文字列を読む。out が非 NULL なら中身を格納する（エスケープは \" \\ のみ想定）。 */
static const char *jstring(const char *p, char *out, size_t cap, jdoc *d)
{
    size_t k = 0;
    if (*p != '"') { jerr(d, p, "文字列の開始 \" が無い"); return p; }
    p++;
    while (*p && *p != '"') {
        char c = *p;
        if (c == '\\') {
            p++;
            if (!*p) break;
            c = *p;
        }
        if (out && k + 1 < cap) out[k++] = c;
        p++;
    }
    if (*p != '"') { jerr(d, p, "文字列が閉じていない"); return p; }
    if (out) out[k] = '\0';
    return p + 1;
}

static void jstore(jdoc *d, const char *path, double v)
{
    if (d->skip) return;
    if (d->n >= JMAX_ENTRIES) { d->ok = 0;
        snprintf(d->err, sizeof(d->err), "JSON 解析エラー: キーが多すぎる（上限 %d）", JMAX_ENTRIES);
        return; }
    snprintf(d->e[d->n].path, JPATH_LEN, "%s", path);
    d->e[d->n].num = v;
    d->n++;
}

static const char *jvalue(const char *p, const char *prefix, jdoc *d, int depth);

static const char *jobject(const char *p, const char *prefix, jdoc *d, int depth)
{
    char key[48];
    char path[JPATH_LEN];

    p++;                              /* '{' */
    p = jskip_ws(p);
    if (*p == '}') return p + 1;

    for (;;) {
        p = jskip_ws(p);
        p = jstring(p, key, sizeof(key), d);
        if (!d->ok) return p;
        p = jskip_ws(p);
        if (*p != ':') { jerr(d, p, "キーの後に : が無い"); return p; }
        p++;
        if (prefix[0]) snprintf(path, sizeof(path), "%s.%s", prefix, key);
        else           snprintf(path, sizeof(path), "%s", key);
        p = jvalue(p, path, d, depth + 1);
        if (!d->ok) return p;
        p = jskip_ws(p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') return p + 1;
        jerr(d, p, "オブジェクト内で , か } を期待した");
        return p;
    }
}

/* 配列は読み飛ばす。params.json では表示用の名前しか入っていない。 */
static const char *jarray(const char *p, const char *prefix, jdoc *d, int depth)
{
    p++;                              /* '[' */
    p = jskip_ws(p);
    if (*p == ']') return p + 1;
    d->skip++;
    for (;;) {
        p = jvalue(p, prefix, d, depth + 1);
        if (!d->ok) break;
        p = jskip_ws(p);
        if (*p == ',') { p++; continue; }
        if (*p == ']') { p++; break; }
        jerr(d, p, "配列内で , か ] を期待した");
        break;
    }
    d->skip--;
    return p;
}

static const char *jvalue(const char *p, const char *path, jdoc *d, int depth)
{
    if (depth > 8) { jerr(d, p, "ネストが深すぎる"); return p; }
    p = jskip_ws(p);
    if (*p == '{') return jobject(p, path, d, depth);
    if (*p == '[') return jarray(p, path, d, depth);
    if (*p == '"') return jstring(p, NULL, 0, d);      /* 文字列値は捨てる */
    if (!strncmp(p, "true", 4))  { jstore(d, path, 1.0); return p + 4; }
    if (!strncmp(p, "false", 5)) { jstore(d, path, 0.0); return p + 5; }
    if (!strncmp(p, "null", 4))  { return p + 4; }
    {
        char *end = NULL;
        double v = strtod(p, &end);
        if (end == p) { jerr(d, p, "値として解釈できない文字"); return p; }
        jstore(d, path, v);
        return end;
    }
}

static int jparse(jdoc *d, const char *text)
{
    const char *p;
    d->n = 0; d->ok = 1; d->skip = 0; d->err[0] = '\0'; d->base = text;
    p = jskip_ws(text);
    if (*p != '{') { jerr(d, p, "先頭が { でない"); return 0; }
    p = jobject(p, "", d, 0);
    if (!d->ok) return 0;
    p = jskip_ws(p);
    if (*p != '\0') { jerr(d, p, "末尾に余分な文字がある"); return 0; }
    return 1;
}

static int jfind(const jdoc *d, const char *path, double *out)
{
    int i;
    for (i = 0; i < d->n; i++) {
        if (!strcmp(d->e[i].path, path)) { *out = d->e[i].num; return 1; }
    }
    return 0;
}

/* ================================================================== */
/* params.json の読み込み                                              */
/* ================================================================== */

typedef struct {
    const jdoc *d;
    char       *err;
    size_t      errcap;
    int         miss;
} preader;

static double req(preader *R, const char *path)
{
    double v = 0.0;
    if (!jfind(R->d, path, &v)) {
        if (!R->miss)
            snprintf(R->err, R->errcap, "params.json に必須キー \"%s\" が無い", path);
        R->miss++;
    }
    return v;
}

static double opt(preader *R, const char *path, double dflt)
{
    double v = 0.0;
    if (!jfind(R->d, path, &v)) return dflt;
    return v;
}

int twin_params_load(twin_params *p, const char *path, char *err, size_t errcap)
{
    static char  text[1 << 16];       /* params.json は小さいので固定バッファ */
    static jdoc  doc;
    preader  R;
    FILE    *f;
    size_t   n;
    double   stations;

    if (err && errcap) err[0] = '\0';
    memset(p, 0, sizeof(*p));

    f = fopen(path, "rb");
    if (!f) {
        snprintf(err, errcap, "params.json を開けない: %s", path);
        return 1;
    }
    n = fread(text, 1, sizeof(text) - 1, f);
    if (ferror(f)) { fclose(f); snprintf(err, errcap, "params.json の読み込みに失敗: %s", path); return 1; }
    if (n >= sizeof(text) - 1) { fclose(f); snprintf(err, errcap, "params.json が大きすぎる（上限 %u バイト）", (unsigned)sizeof(text) - 1); return 1; }
    text[n] = '\0';
    fclose(f);

    if (!jparse(&doc, text)) { snprintf(err, errcap, "%s", doc.err); return 1; }

    R.d = &doc; R.err = err; R.errcap = errcap; R.miss = 0;

    /* ボトル（mm -> m）。肩・首は外形だけの情報なので液の計算には使わない。 */
    p->bottle_R     = req(&R, "bottle.inner_diameter_mm") * 0.5e-3;
    p->body_height  = req(&R, "bottle.body_height_mm")    * 1e-3;
    p->bottle_mass  = req(&R, "bottle.empty_mass_kg");

    /* 液 */
    p->rho  = req(&R, "liquid.density_kg_m3");
    p->nu   = opt(&R, "liquid.kinematic_viscosity_m2_s", 0.0);
    p->zeta = req(&R, "liquid.slosh_damping_ratio");

    /* テーブル */
    p->Rp           = req(&R, "table.pitch_circle_diameter_mm") * 0.5e-3;
    stations        = req(&R, "table.stations");
    p->index_angle  = req(&R, "table.index_angle_deg") * (M_PI / 180.0);
    p->bare_inertia = req(&R, "table.bare_inertia_kg_m2");
    p->n_stations   = (int)(stations + 0.5);
    p->plate_diameter  = opt(&R, "table.plate_diameter_mm", 0.0) * 1e-3;
    p->plate_thickness = opt(&R, "table.plate_thickness_mm", 0.0) * 1e-3;
    p->plate_density   = opt(&R, "table.plate_density_kg_m3", 0.0);
    p->plate_mass      = opt(&R, "table.plate_mass_kg", 0.0);

    /* ステーション（世界角） */
    p->infeed_angle    = req(&R, "stations.infeed_deg")    * (M_PI / 180.0);
    p->fill_angle      = req(&R, "stations.fill_deg")      * (M_PI / 180.0);
    p->discharge_angle = req(&R, "stations.discharge_deg") * (M_PI / 180.0);

    /* カム索引ユニット。摩擦の 3 つは出所が無い仮置き（_friction_grade: assumed）。 */
    p->index_angle_input = req(&R, "indexer.index_angle_input_deg") * (M_PI / 180.0);
    p->dwell_angle_input = req(&R, "indexer.dwell_angle_input_deg") * (M_PI / 180.0);
    p->cam_efficiency    = req(&R, "indexer.efficiency");
    p->input_drag_torque = req(&R, "indexer.input_drag_torque_Nm");
    p->table_viscous     = req(&R, "indexer.table_viscous_Nms_rad");

    /* サイクル。dwell_time_s は冗長なので tact - index_time で導出する。 */
    p->tact       = req(&R, "cycle.tact_s");
    p->index_time = req(&R, "cycle.index_time_s");
    p->dwell      = p->tact - p->index_time;

    /* 駆動系 */
    p->gear_ratio  = req(&R, "drive.gear_ratio");
    p->motor_speed = req(&R, "drive.motor_rpm_at_operating_point") * (2.0 * M_PI / 60.0);

    /* 充填 */
    p->target_volume    = req(&R, "fill.target_volume_mL") * 1e-6;
    p->flow_rate        = req(&R, "fill.flow_rate_mL_s")   * 1e-6;
    p->nozzle_diameter  = req(&R, "fill.nozzle_diameter_mm") * 1e-3;
    p->valve_open_delay = req(&R, "fill.valve_open_delay_s");
    p->valve_close_delay= req(&R, "fill.valve_close_delay_s");
    p->start_delay      = req(&R, "fill.start_delay_s");

    /* sim */
    p->dt         = req(&R, "sim.dt_s");
    p->log_rate   = req(&R, "sim.log_rate_hz");
    p->frame_rate = opt(&R, "sim.frame_rate_hz", 30.0);
    p->g          = req(&R, "sim.gravity_m_s2");

    /* 取り決め: ノズル出口から液面までの落差は params.json に無い。既定 0。 */
    p->jet_fall_height = 0.0;
    p->no_coriolis     = 0;

    /* 故障モード（無い場合は無効）。 */
    p->flt.missing_enabled = (opt(&R, "faults.missing_bottle.enabled", 0.0) != 0.0);
    p->flt.missing_station = (int)(opt(&R, "faults.missing_bottle.station", 0.0) + 0.5);

    p->flt.valve_enabled     = (opt(&R, "faults.valve_close_delay.enabled", 0.0) != 0.0);
    p->flt.valve_extra_delay =  opt(&R, "faults.valve_close_delay.extra_delay_s", 0.0);
    p->flt.valve_drip_volume =  opt(&R, "faults.valve_close_delay.drip_volume_mL", 0.0) * 1e-6;

    p->flt.cam_enabled       = (opt(&R, "faults.cam_follower_wear.enabled", 0.0) != 0.0);
    p->flt.cam_clearance     =  opt(&R, "faults.cam_follower_wear.clearance_deg", 0.0) * (M_PI / 180.0);
    p->flt.cam_impact_torque =  opt(&R, "faults.cam_follower_wear.impact_torque_Nm", 0.0);

    p->flt.bear_enabled      = (opt(&R, "faults.bearing_outer_race.enabled", 0.0) != 0.0);
    p->flt.bear_defect_freq  =  opt(&R, "faults.bearing_outer_race.defect_freq_hz", 0.0);
    p->flt.bear_accel        =  opt(&R, "faults.bearing_outer_race.impulse_accel_m_s2", 0.0);
    p->flt.bear_ring_freq    =  opt(&R, "faults.bearing_outer_race.ring_freq_hz", 0.0);
    p->flt.bear_ring_damping =  opt(&R, "faults.bearing_outer_race.ring_damping", 0.0);

    if (R.miss) return 1;

    /* 妥当性 */
    if (p->n_stations < 1 || p->n_stations > TWIN_MAX_STATIONS) {
        snprintf(err, errcap, "table.stations が範囲外: %d（1..%d）", p->n_stations, TWIN_MAX_STATIONS);
        return 1;
    }
    if (!(p->dt > 0.0))       { snprintf(err, errcap, "sim.dt_s が正でない"); return 1; }
    if (!(p->log_rate > 0.0)) { snprintf(err, errcap, "sim.log_rate_hz が正でない"); return 1; }
    if (!(p->bottle_R > 0.0)) { snprintf(err, errcap, "bottle.inner_diameter_mm が正でない"); return 1; }
    if (!(p->tact > 0.0))     { snprintf(err, errcap, "cycle.tact_s が正でない"); return 1; }
    if (!(p->gear_ratio > 0.0)) { snprintf(err, errcap, "drive.gear_ratio が正でない"); return 1; }
    if (!(p->index_time > 0.0) || p->index_time > p->tact) {
        snprintf(err, errcap, "cycle.index_time_s が範囲外（0 < index_time <= tact）"); return 1;
    }
    if (!(p->index_angle_input > 0.0)) {
        snprintf(err, errcap, "indexer.index_angle_input_deg が正でない"); return 1;
    }
    if (!(p->cam_efficiency > 0.0) || p->cam_efficiency > 1.0) {
        snprintf(err, errcap, "indexer.efficiency が範囲外（0 < eta <= 1）"); return 1;
    }
    if (p->flt.missing_station < 0 || p->flt.missing_station >= p->n_stations) {
        snprintf(err, errcap, "faults.missing_bottle.station が範囲外: %d（0..%d）",
                 p->flt.missing_station, p->n_stations - 1);
        return 1;
    }
    /* 工程の世界角はステーション間隔の整数倍でなければ、そこにボトルが来ない。 */
    {
        const double pitch = 2.0 * M_PI / (double)p->n_stations;
        const char *nm[3] = { "stations.infeed_deg", "stations.fill_deg", "stations.discharge_deg" };
        double ang[3];
        int i;
        ang[0] = p->infeed_angle; ang[1] = p->fill_angle; ang[2] = p->discharge_angle;
        for (i = 0; i < 3; i++) {
            double kk = ang[i] / pitch;
            if (fabs(kk - floor(kk + 0.5)) > 1e-9) {
                snprintf(err, errcap,
                         "%s (%.4f deg) がステーション間隔 %.4f deg の整数倍でない",
                         nm[i], ang[i] * 180.0 / M_PI, pitch * 180.0 / M_PI);
                return 1;
            }
        }
        if (p->infeed_angle == p->discharge_angle) {
            snprintf(err, errcap,
                     "供給と排出が同じ世界角にある。同じポケットで同時に抜いて入れることはできない");
            return 1;
        }
    }
    return 0;
}

/* ================================================================== */
/* 幾何・派生量（params.py のプロパティに対応）                        */
/* ================================================================== */

static double cross_section(const twin_params *p)
{
    return M_PI * p->bottle_R * p->bottle_R;
}

static double station_pitch(const twin_params *p)
{
    return 2.0 * M_PI / (double)p->n_stations;
}

static double input_shaft_speed(const twin_params *p)
{
    return 2.0 * M_PI / p->tact;      /* 1 タクトで入力軸 1 回転 */
}

static double nozzle_velocity(const twin_params *p)
{
    double a = M_PI * (p->nozzle_diameter / 2.0) * (p->nozzle_diameter / 2.0);
    return p->flow_rate / a;
}

static double station_world_angle(const twin_params *p, int i, double th_t)
{
    return th_t + (double)i * station_pitch(p);
}

/* 世界角 world にいるホルダ番号。
 * 丸めは nearbyint（既定の丸めモード = 偶数丸め）で、Python の round と揃えてある。 */
int twin_holder_at(const twin_params *p, double world, double th_t)
{
    double k = (world - th_t) / station_pitch(p);
    int    i = (int)nearbyint(k);
    int    n = p->n_stations;
    return ((i % n) + n) % n;
}

/* ================================================================== */
/* 1. カム曲線（変形正弦）                                             */
/* ================================================================== */
/*
 * 加速度を 3 区間の正弦でつないだ曲線。無次元時間 x = 0..1 に対し
 *   0   .. 1/8 : a = Ca*sin(4pi*x)
 *   1/8 .. 7/8 : a = Ca*cos(4pi(x-1/8)/3)
 *   7/8 .. 1   : a = -Ca*cos(4pi(x-7/8))
 * 変位が 0 から 1 になるよう振幅を決めると、その振幅がそのまま無次元最大加速度
 *   Ca = 4*pi^2/(pi+4) = 5.52796,  Cv = Ca/pi = 1.75960
 * になる。両端で 変位 0/1、速度 0、加速度 0。だから停留と滑らかにつながる。
 */

void twin_modified_sine(double x, double *sp, double *vp, double *ap)
{
    const double A  = 4.0 * M_PI * M_PI / (M_PI + 4.0);
    const double v1 = A / (4.0 * M_PI);
    const double s1 = v1 * (1.0 / 8.0 - 1.0 / (4.0 * M_PI));
    const double s2 = s1 + v1 * 0.75 + 9.0 * A / (8.0 * M_PI * M_PI);

    if (x <= 0.0) { *sp = 0.0; *vp = 0.0; *ap = 0.0; return; }
    if (x >= 1.0) { *sp = 1.0; *vp = 0.0; *ap = 0.0; return; }

    if (x < 1.0 / 8.0) {
        double u = x;
        *sp = v1 * (u - sin(4.0 * M_PI * u) / (4.0 * M_PI));
        *vp = v1 * (1.0 - cos(4.0 * M_PI * u));
        *ap = A * sin(4.0 * M_PI * u);
        return;
    }
    if (x < 7.0 / 8.0) {
        double u = x - 1.0 / 8.0;
        double w = 4.0 * M_PI / 3.0;
        *sp = s1 + v1 * u + (9.0 * A / (16.0 * M_PI * M_PI)) * (1.0 - cos(w * u));
        *vp = v1 + (3.0 * A / (4.0 * M_PI)) * sin(w * u);
        *ap = A * cos(w * u);
        return;
    }
    {
        double u = x - 7.0 / 8.0;
        *sp = s2 + v1 * u + (A / (16.0 * M_PI * M_PI)) * (cos(4.0 * M_PI * u) - 1.0);
        *vp = v1 - (A / (4.0 * M_PI)) * sin(4.0 * M_PI * u);
        *ap = -A * cos(4.0 * M_PI * u);
    }
}

/* 時刻 t のカム入力軸角 [rad]（0 <= psi < 2pi に畳む）。
 * **サイクル内時刻から作ること。** 通し時刻に剰余を取ると、サイクルの境目で
 * 丸め次第で 2pi の直前に落ちて、テーブル角が 1 ステップだけ飛ぶ。 */
static double cam_input_angle(const twin_params *p, double t_in_cycle)
{
    double two_pi = 2.0 * M_PI;
    double psi = fmod(input_shaft_speed(p) * t_in_cycle, two_pi);
    if (psi < 0.0) psi += two_pi;
    return psi;
}

/* 割出し区間の進行度 x = 0..1。停留区間では 1 を返す。 */
static double cam_progress(const twin_params *p, double psi)
{
    if (psi <= 0.0) return 0.0;
    if (psi >= p->index_angle_input) return 1.0;
    return psi / p->index_angle_input;
}

static int is_index_phase(const twin_params *p, double psi)
{
    return (psi >= 0.0 && psi < p->index_angle_input);
}

void twin_table_motion(const twin_params *p, double psi,
                       double *th, double *omega, double *alpha)
{
    double x = cam_progress(p, psi);
    double s, v, a, rate;
    twin_modified_sine(x, &s, &v, &a);
    rate = 1.0 / p->index_time;           /* dx/dt [1/s]。入力軸が一定速なので定数 */
    *th    = p->index_angle * s;
    *omega = p->index_angle * v * rate;
    *alpha = p->index_angle * a * rate * rate;
}

double twin_table_ratio(const twin_params *p, double psi)
{
    double x = cam_progress(p, psi);
    double s, v, a;
    twin_modified_sine(x, &s, &v, &a);
    (void)s; (void)a;
    return p->index_angle * v / p->index_angle_input;
}

/* ================================================================== */
/* 2. スロッシング（回転座標系・コリオリ結合）                         */
/* ================================================================== */

double twin_slosh_omega(const twin_params *p, double h)
{
    double w1sq;
    if (h <= 0.0) return 0.0;
    w1sq = (p->g * TWIN_EPS1 / p->bottle_R) * tanh(TWIN_EPS1 * h / p->bottle_R);
    return sqrt(w1sq);
}

static double pendulum_length(const twin_params *p, double w1)
{
    if (w1 <= 0.0) return 0.0;        /* 呼ぶ側で w1 <= 0 を弾く。ref.py は inf */
    return p->g / (w1 * w1);
}

/* 液のうち揺れる分の割合 m1/m_liq。
 *   m1/m_liq = 2R*tanh(eps1*h/R) / ( eps1*(eps1^2-1)*h )
 * 残り m0 = m_liq - m1 は容器と一緒に動く「凍った液」として扱う。 */
double twin_slosh_mass_ratio(const twin_params *p, double h)
{
    double denom;
    if (h <= 0.0) return 0.0;
    denom = TWIN_EPS1 * (TWIN_EPS1 * TWIN_EPS1 - 1.0) * h;
    return 2.0 * p->bottle_R * tanh(TWIN_EPS1 * h / p->bottle_R) / denom;
}

/* 振り子を 1 ステップ進める（半陰的オイラー）。
 *
 * 回転する基底 r^, t^ で書くと、基底の回転ぶんの 3 項が出る。
 *   phi_r'' =  2*om*phi_t' + al*phi_t + om^2*phi_r - 2*zeta*w1*phi_r' - w1^2*phi_r - a_r/L1
 *   phi_t'' = -2*om*phi_r' - al*phi_r + om^2*phi_t - 2*zeta*w1*phi_t' - w1^2*phi_t - a_t/L1
 * 2*om*phi' がコリオリ、al*phi がオイラー、om^2*phi が遠心。
 * 結合項は今のステップの速度を使う（陽的）。
 *
 * 減衰は容器に対する相対速度（回転座標系での微分）に掛けている。減衰の実体は
 * 壁の境界層なので相対のほうが素直で、この選び方だと反力の式が厳密に閉じる。 */
static void step_slosh(twin_state *s, int i, double a_t, double a_r,
                       double omega, double alpha, double w1, double dt)
{
    const twin_params *p = &s->p;
    double L1, ddphi_r, ddphi_t;
    if (w1 <= 0.0) return;
    L1 = pendulum_length(p, w1);

    /* 比較用。落とすと前版と同じ「2 方向が独立な振り子」に戻る。既定は入れる側。 */
    if (p->no_coriolis) { omega = 0.0; alpha = 0.0; }

    ddphi_r = (2.0 * omega * s->dphi_t[i]
               + alpha * s->phi_t[i]
               + omega * omega * s->phi_r[i]
               - 2.0 * p->zeta * w1 * s->dphi_r[i]
               - w1 * w1 * s->phi_r[i]
               - a_r / L1);

    ddphi_t = (-2.0 * omega * s->dphi_r[i]
               - alpha * s->phi_r[i]
               + omega * omega * s->phi_t[i]
               - 2.0 * p->zeta * w1 * s->dphi_t[i]
               - w1 * w1 * s->phi_t[i]
               - a_t / L1);

    s->dphi_r[i] = s->dphi_r[i] + ddphi_r * dt;
    s->dphi_t[i] = s->dphi_t[i] + ddphi_t * dt;
    s->phi_r[i]  = s->phi_r[i] + s->dphi_r[i] * dt;
    s->phi_t[i]  = s->phi_t[i] + s->dphi_t[i] * dt;
}

/* 揺れる質量 m1 の慣性系での加速度。
 *   a1 = -( g*phi + 2*zeta*w1*L1*phi' )
 * 振り子の運動方程式を代入すると容器の加速度が消えてこの形になる（w1^2*L1 = g）。 */
static void slosh_accel(const twin_state *s, int i, double w1, double L1,
                        double *a1_t, double *a1_r)
{
    const twin_params *p = &s->p;
    *a1_t = -(p->g * s->phi_t[i] + 2.0 * p->zeta * w1 * L1 * s->dphi_t[i]);
    *a1_r = -(p->g * s->phi_r[i] + 2.0 * p->zeta * w1 * L1 * s->dphi_r[i]);
}

/* 液がボトルに及ぼす水平力（局所座標）。**返すのは反作用のほう。**
 *   F_hold  =   m0*a_tank + m1*a1      ホルダが中身に及ぼす力（液を動かす力）
 *   F_react = -(m0*a_tank + m1*a1)     中身がホルダに返す力。返り値はこちら
 * 符号は MODEL.md 2.4 の「テーブルが受ける側」に揃えてある。T_slosh と同じ向き。
 * m0 の分は J_load 側にも入っているので、軸トルクに足すときは m1 の分だけにすること。 */
static void liquid_force(const twin_state *s, int i, double volume,
                         double a_t, double a_r, double w1, double L1,
                         double *f_t, double *f_r)
{
    const twin_params *p = &s->p;
    double h, m_liq, m1, m0, a1_t, a1_r;
    if (volume <= 0.0) { *f_t = 0.0; *f_r = 0.0; return; }
    h = volume / cross_section(p);
    m_liq = p->rho * volume;
    m1 = m_liq * twin_slosh_mass_ratio(p, h);
    m0 = m_liq - m1;
    slosh_accel(s, i, w1, L1, &a1_t, &a1_r);
    *f_t = -(m0 * a_t + m1 * a1_t);
    *f_r = -(m0 * a_r + m1 * a1_r);
}

/* こぼれを判定して (残った体積, こぼれた体積) を返す。
 *
 * h + dz が body_height を超えたら、超過分の高さに断面積を掛けたものを引く。
 * 実際に縁を越えるのは傾いた液面が縁を切る楔だけなので、この扱いはこぼれを
 * 多めに見積もる（py/NOTES.md 6 節。1〜2 桁の過大評価）。
 * **柱のままにしてあるのは py/ref.py が柱だから。** 楔に直すなら両方直すこと。
 * この諸元では傾き 68 mrad に対し頭上空間 29.5 mm あるので、そもそも届かない。 */
static double apply_spill(const twin_params *p, double volume, double tilt,
                          double *spilled)
{
    double A = cross_section(p);
    double h, dz, over, sp;
    *spilled = 0.0;
    if (volume <= 0.0) return 0.0;
    h = volume / A;
    dz = p->bottle_R * tan(tilt);     /* tilt が pi/2 に近いところは近似の外。値を信じない */
    over = h + dz - p->body_height;
    if (over <= 0.0) return volume;
    sp = over * A;
    if (sp > volume) sp = volume;
    *spilled = sp;
    return volume - sp;
}

/* ================================================================== */
/* 3. 慣性と質量                                                       */
/* ================================================================== */

/* ホルダ 1 個に載っている質量（液は全量）。曲げモーメントで使う。 */
static double holder_mass(const twin_state *s, int i)
{
    const twin_params *p = &s->p;
    if (!s->has_bottle[i]) return 0.0;
    return p->bottle_mass + p->rho * s->V[i];
}

/* ホルダ 1 個の質量のうち、剛体として一緒に回る分。
 * 液の揺れる分 m1 は振り子として別に動くので抜いてある。 */
static double rigid_mass(const twin_state *s, int i)
{
    const twin_params *p = &s->p;
    double h, m_liq, m1;
    if (!s->has_bottle[i]) return 0.0;
    if (s->V[i] <= 0.0) return p->bottle_mass;
    h = s->V[i] / cross_section(p);
    m_liq = p->rho * s->V[i];
    m1 = m_liq * twin_slosh_mass_ratio(p, h);
    return p->bottle_mass + m_liq - m1;
}

/* 液の m1 を抜いた J_load。反力トルク（T_slosh）と組にして使う。 */
static double rigid_load_inertia(const twin_state *s)
{
    const twin_params *p = &s->p;
    double j = p->bare_inertia;
    int i;
    for (i = 0; i < p->n_stations; i++)
        j = j + rigid_mass(s, i) * p->Rp * p->Rp;
    return j;
}

/* ================================================================== */
/* 4. 充填                                                             */
/* ================================================================== */

/* 充填流がテーブルに与える鉛直下向きの力。
 *   F = rho*Q*v_impact,  v_impact = sqrt(v_nozzle^2 + 2*g*fall_height)
 * 落差は params.json に無い。既定 0（ノズル出口の流速がそのまま着液速度）。 */
static double jet_force(const twin_params *p)
{
    double fall = p->jet_fall_height > 0.0 ? p->jet_fall_height : 0.0;
    double vn = nozzle_velocity(p);
    double v_impact = sqrt(vn * vn + 2.0 * p->g * fall);
    return p->rho * p->flow_rate * v_impact;
}

/* 弁を閉じろと命じる体積。閉じ遅れを先読みして早めに命じる、という取り決め
 * （params.json は閉じ命令の出し方を決めていない）。
 *
 * **先読みするのは公称の遅れ（fill.valve_close_delay_s）だけ。** 制御側は弁が
 * 劣化して遅くなったことを知らないので、故障ぶんの遅れは先読みできない。
 * だから正常時はちょうど target_volume で止まり、閉じ遅れ故障では
 * flow_rate * extra_delay_s だけ過充填になる。 */
static double close_command_volume(const twin_params *p)
{
    return p->target_volume - p->flow_rate * p->valve_close_delay;
}

/* 揺れる質量 m1 の体積についての微係数 dm1/dV [kg/m^3]。
 *   m1     = rho * 2*R*A*tanh(eps1*h/R) / (eps1*(eps1^2-1))     （h = V/A）
 *   dm1/dV = 2*rho / ( (eps1^2 - 1) * cosh^2(eps1*h/R) )
 * V が消えた形になるので厳密に微分できる。数値差分は取らない。 */
static double slosh_mass_rate(const twin_params *p, double volume)
{
    double h = volume / cross_section(p);
    double c = cosh(TWIN_EPS1 * h / p->bottle_R);
    return 2.0 * p->rho / ((TWIN_EPS1 * TWIN_EPS1 - 1.0) * c * c);
}

/* 充填中の dJ_load/dt [kg m^2/s]。液は m0 だけ数える側の J_load の時間微分。
 *   dJ_load/dt = Rp^2 * (dm0/dV) * flow_rate,   dm0/dV = rho - dm1/dV
 * ノズルは世界座標に固定なので、入ってくる液の接線速度は 0。角運動量の収支から
 * テーブル軸には (dJ_load/dt)*omega の項が要る（MODEL.md 7.4）。
 * **公称条件ではこの項は厳密に 0**（充填は停留中に終わり、停留では omega = 0）。
 * それでも式には入れる。タクトを詰めて充填が割出しにはみ出すと効き始めるため。 */
static double load_inertia_rate(const twin_params *p, double volume)
{
    return (p->rho - slosh_mass_rate(p, volume)) * p->flow_rate * p->Rp * p->Rp;
}

/* ================================================================== */
/* 5. 垂直荷重経路                                                     */
/* ================================================================== */

/* ボトル重量（＋充填中は噴流）によるテーブル軸の曲げモーメント。
 *   M_x = Rp * sum( F_i*cos(ang_i) ),  M_y = Rp * sum( F_i*sin(ang_i) )
 * 全ホルダに同じ質量が載っていれば 8 方位の和が 0 になって消える。
 * 残るのは中身の偏りだけ。**どこで受けるかは決めない**（FORMAT.md 2.7）。 */
static void bending_moment(const twin_state *s, double th_t,
                           int jet_station, double jet_F,
                           double *mx, double *my, double *mabs)
{
    const twin_params *p = &s->p;
    double sx = 0.0, sy = 0.0, ang, force;
    int i;
    for (i = 0; i < p->n_stations; i++) {
        ang = station_world_angle(p, i, th_t);
        force = holder_mass(s, i) * p->g;
        sx = sx + force * cos(ang);
        sy = sy + force * sin(ang);
    }
    if (jet_station >= 0) {
        ang = station_world_angle(p, jet_station, th_t);
        sx = sx + jet_F * cos(ang);
        sy = sy + jet_F * sin(ang);
    }
    *mx = p->Rp * sx;
    *my = p->Rp * sy;
    *mabs = hypot(*mx, *my);
}

/* ================================================================== */
/* 6. イベント                                                         */
/* ================================================================== */

static void ev_zero(twin_event *e)
{
    int i;
    e->t = 0.0; e->th_t = 0.0; e->th_m = 0.0;
    e->kind = -1; e->station = -1;
    for (i = 0; i < TWIN_EV_DATA; i++) e->d[i] = 0.0;
}

/* 発生したイベントを 1 件積む。溢れたら数だけ数えて捨てる。 */
static twin_event *push_event(twin_state *s, int kind, int station)
{
    twin_event *e;
    if (s->n_ev >= TWIN_MAX_EVENTS) { s->n_ev_lost++; return NULL; }
    e = &s->ev[s->n_ev++];
    ev_zero(e);
    e->kind = kind;
    e->station = station;
    e->t = s->t;
    e->th_t = s->th_t;
    e->th_m = s->th_m;
    s->n_ev_total++;
    return e;
}

/* 予定イベント（サイクル頭で作る）のうち、時刻 t までのものを出す。 */
static void flush_sched(twin_state *s, double t)
{
    while (s->i_sched < s->n_sched && s->sched[s->i_sched].t <= t) {
        if (s->n_ev >= TWIN_MAX_EVENTS) { s->n_ev_lost++; s->i_sched++; continue; }
        s->ev[s->n_ev++] = s->sched[s->i_sched++];
        s->n_ev_total++;
    }
}

/* このサイクルの予定イベントを作る。
 * py/ref.py はサイクル頭で一括して append し、最後に時刻で安定ソートしている。
 * 同じ時刻ならカムフォロワが先、軸受が後になるので、その順で作ってから
 * 安定な挿入ソートを掛ける（並びまで揃えないと件数以外が比べられない）。 */
static void build_schedule(twin_state *s)
{
    const twin_params *p = &s->p;
    const double t0 = s->t0;
    twin_event *e;
    int i, j;

    s->n_sched = 0;
    s->i_sched = 0;

    /* カムフォロワの当たり（割出しの入口と出口） */
    if (p->flt.cam_enabled) {
        double hit[2];
        hit[0] = t0;
        hit[1] = t0 + p->index_time;
        for (i = 0; i < 2; i++) {
            if (s->n_sched >= TWIN_MAX_SCHED) break;
            e = &s->sched[s->n_sched++];
            ev_zero(e);
            e->kind = TWIN_EV_CAM_IMPACT;
            e->station = -1;
            e->t = hit[i];
            e->d[0] = p->flt.cam_impact_torque;
            e->d[1] = p->flt.cam_clearance;
        }
    }

    /* 軸受外輪傷（モータ軸）。入力軸が連続回転するので、テーブルが止まっている
     * 停留中も等間隔で出続ける。1 タクトあたり 268 件。 */
    if (p->flt.bear_enabled && p->flt.bear_defect_freq > 0.0) {
        double period = 1.0 / p->flt.bear_defect_freq;
        double t1 = t0 + p->tact;
        long k = (long)ceil(t0 / period);
        while ((double)k * period < t1) {
            double th;
            if (s->n_sched >= TWIN_MAX_SCHED) { s->n_ev_lost++; break; }
            e = &s->sched[s->n_sched++];
            ev_zero(e);
            e->kind = TWIN_EV_BEARING;
            e->station = -1;
            e->t = (double)k * period;
            e->d[0] = p->flt.bear_accel;
            e->d[1] = p->flt.bear_ring_freq;
            e->d[2] = p->flt.bear_ring_damping;
            /* 停留中に出るものが多いので、テーブル角は「その時刻の値」を入れる。
             * 割出し中なら動いている途中の角になる。 */
            {
                double tin = e->t - t0;
                double psi = cam_input_angle(p, tin);
                double om, al;
                twin_table_motion(p, psi, &th, &om, &al);
                e->th_t = s->th_base + th;
            }
            e->th_m = p->motor_speed * e->t;
            k++;
        }
    }

    /* 時刻で安定な挿入ソート（同時刻はカムフォロワが先） */
    for (i = 1; i < s->n_sched; i++) {
        twin_event tmp = s->sched[i];
        j = i - 1;
        while (j >= 0 && s->sched[j].t > tmp.t) {
            s->sched[j + 1] = s->sched[j];
            j--;
        }
        s->sched[j + 1] = tmp;
    }
}

/* ================================================================== */
/* 初期化                                                              */
/* ================================================================== */

/* 液量から決まる量を作り直す（液深・固有角周波数・等価振り子長・等価質量）。 */
static void update_liquid(twin_state *s)
{
    const twin_params *p = &s->p;
    const double A = cross_section(p);
    int i;
    for (i = 0; i < p->n_stations; i++) {
        double h = s->V[i] / A;
        double w1 = twin_slosh_omega(p, h);
        double m_liq = p->rho * s->V[i];
        double m1 = m_liq * twin_slosh_mass_ratio(p, h);
        s->h[i]  = h;
        s->w1[i] = w1;
        s->L1[i] = pendulum_length(p, w1);
        s->m1[i] = m1;
        s->m0[i] = m_liq - m1;
    }
    s->j_load = rigid_load_inertia(s);
}

void twin_refresh(twin_state *s)
{
    update_liquid(s);
}

void twin_init(twin_state *s, const twin_params *p, int prime)
{
    int i;

    memset(s, 0, sizeof(*s));
    s->p = *p;

    for (i = 0; i < TWIN_MAX_STATIONS; i++) {
        s->has_bottle[i] = 0;
        s->filled[i] = 0;
        s->V[i] = s->spill[i] = 0.0;
        s->phi_t[i] = s->phi_r[i] = s->dphi_t[i] = s->dphi_r[i] = 0.0;
        s->h[i] = s->w1[i] = s->L1[i] = s->m0[i] = s->m1[i] = 0.0;
        s->dzr[i] = 0.0;
        s->range_bad[i] = 0;
    }

    /* 初期状態（FORMAT.md 4 節）
     *
     * TWIN_PRIME_GEOM（既定）: 工程配置から作る。テーブル角 0 で、ホルダ i の
     *   世界角は i*pitch。供給からの割出し回数を k = (i*pitch - infeed)/pitch mod N、
     *   滞留を n_res = (discharge - infeed)/pitch mod N とすると
     *     k == 0        : 空のボトル（供給を受けたところ）
     *     0 < k < n_res : 満量（充填ステーションを過ぎている）
     *     k >= n_res    : ボトル無し（排出済みで、まだ供給に来ていない）
     *   供給 315 / 充填 0 / 排出 225 なら n_res = 6 で
     *   **満量 5 本 + 空瓶 1 本 + 空ホルダ 2 つ**。
     *   py/ref.py の steady_holders と同じ式（2026-07-23 に向こうも幾何から作る形になった。
     *   それまでの互換モード --prime-ref は要らなくなったので撤去した）。 */
    if (prime == TWIN_PRIME_GEOM) {
        double pitch = station_pitch(p);
        int n_res = (int)nearbyint((p->discharge_angle - p->infeed_angle) / pitch);
        n_res = ((n_res % p->n_stations) + p->n_stations) % p->n_stations;
        for (i = 0; i < p->n_stations; i++) {
            int k = (int)nearbyint(((double)i * pitch - p->infeed_angle) / pitch);
            k = ((k % p->n_stations) + p->n_stations) % p->n_stations;
            if (k >= n_res) continue;                 /* 排出済み。ボトルが無い */
            s->has_bottle[i] = 1;
            s->V[i] = (k == 0) ? 0.0 : p->target_volume;
            s->filled[i] = (k == 0) ? 0 : 1;
        }
    }

    s->step = 0;
    s->k = 0;
    s->cycle = 0;
    s->t = 0.0;
    s->t0 = 0.0;
    s->th_base = 0.0;
    s->next_log = 0.0;
    s->log_interval = 1.0 / p->log_rate;
    s->do_log = 0;
    s->n_steps_cycle = (long)floor(p->tact / p->dt + 0.5);
    if (s->n_steps_cycle < 1) s->n_steps_cycle = 1;
    s->i_out = s->i_in = -1;
    s->i_fill = -1;
    s->transferred = 0;
    s->closing = 0;
    s->n_ev = 0;
    s->n_sched = 0;
    s->i_sched = 0;
    s->min_t_input = 1.0e300;   /* 符号付きの最小を取るので、0 からは始めない */

    update_liquid(s);
}

/* ================================================================== */
/* dt / log_rate の確認                                                */
/* ================================================================== */

int twin_check_dt(const twin_state *s, char *msg, size_t cap)
{
    const twin_params *p = &s->p;
    double h_full, w1, period, ratio_log;
    int bad = 0;
    size_t used = 0;

    if (msg && cap) msg[0] = '\0';

    /* 満量の液深で固有角周波数は最大、周期は最小になる（tanh が飽和するので実質そこ） */
    h_full = p->target_volume / cross_section(p);
    w1 = twin_slosh_omega(p, h_full);
    period = (w1 > 0.0) ? (2.0 * M_PI / w1) : 0.0;

    if (period > 0.0 && p->dt * 20.0 > period) {
        bad++;
        if (msg && cap > used)
            used += (size_t)snprintf(msg + used, cap - used,
                "警告: dt=%.3g s はスロッシング周期 %.4f s に対して粗い（1 周期あたり %.1f 点）\n",
                p->dt, period, period / p->dt);
    }
    if (p->log_rate * p->dt > 1.0) {
        bad++;
        if (msg && cap > used)
            used += (size_t)snprintf(msg + used, cap - used,
                "警告: log_rate_hz=%.4g は 1/dt=%.4g より速い\n", p->log_rate, 1.0 / p->dt);
    }
    /* ログ間隔が dt の整数倍でないと、間引きの刻みが揺れてスペクトルがずれる */
    ratio_log = 1.0 / (p->log_rate * p->dt);
    if (fabs(ratio_log - floor(ratio_log + 0.5)) > 1e-9) {
        bad++;
        if (msg && cap > used)
            used += (size_t)snprintf(msg + used, cap - used,
                "注意: 1/(log_rate*dt) = %.4f が整数でない。ログの刻みが揺れる\n", ratio_log);
    }
    /* 軸受のリンギングは連続ログには乗せていない（イベント列で渡す）。
     * dt がリンギング周期に対して粗いと、センサ層が合成する時刻の量子化が粗くなる。 */
    if (p->flt.bear_enabled && p->flt.bear_ring_freq > 0.0) {
        if (p->dt > 1.0 / (10.0 * p->flt.bear_ring_freq)) {
            bad++;
            if (msg && cap > used)
                used += (size_t)snprintf(msg + used, cap - used,
                    "注意: dt=%.3g s ではリンギング %.4g Hz を 1 周期あたり %.1f 点でしか刻めない"
                    "（イベントの時刻がその粒度に量子化される）\n",
                    p->dt, p->flt.bear_ring_freq, 1.0 / (p->dt * p->flt.bear_ring_freq));
        }
    }
    (void)used;
    return bad;
}

/* ================================================================== */
/* サイクルの頭で決まること                                            */
/* ================================================================== */

static void begin_cycle(twin_state *s)
{
    const twin_params *p = &s->p;
    double th_now, dwell_start;

    s->t0 = (double)s->cycle * p->tact;

    /* 受け渡しは停留の先頭で起きる。そのときのテーブル角で数える。 */
    th_now = s->th_base + p->index_angle;
    s->i_out = twin_holder_at(p, p->discharge_angle, th_now);
    s->i_in  = twin_holder_at(p, p->infeed_angle, th_now);

    dwell_start = s->t0 + p->index_time;
    s->valve_open_t = dwell_start + p->start_delay + p->valve_open_delay;
    s->closing = 0;
    s->close_t = 0.0;
    s->transferred = 0;

    build_schedule(s);
}

/* ================================================================== */
/* 1 ステップ                                                          */
/* ================================================================== */

void twin_step(twin_state *s)
{
    const twin_params *p = &s->p;
    const double dt = p->dt;
    const double Rp = p->Rp;
    const double A  = cross_section(p);
    double t_in_cycle, th_rel;
    double f_world_x = 0.0, f_world_y = 0.0, torque_slosh = 0.0;
    double extra_close_delay = p->flt.valve_enabled ? p->flt.valve_extra_delay : 0.0;
    double drip_volume = p->flt.valve_enabled ? p->flt.valve_drip_volume : 0.0;
    int missing = p->flt.missing_enabled ? (p->flt.missing_station % p->n_stations) : -1;
    int i, jet_station = -1;
    double jet_F = 0.0;

    s->n_ev = 0;
    if (s->k == 0) begin_cycle(s);

    /* --- 時刻と割出しの運動 ---------------------------------------- */
    t_in_cycle = (double)s->k * dt;
    s->t = s->t0 + t_in_cycle;
    flush_sched(s, s->t);

    s->psi = cam_input_angle(p, t_in_cycle);
    s->in_index = is_index_phase(p, s->psi);
    twin_table_motion(p, s->psi, &th_rel, &s->omega, &s->alpha);
    s->th_t = s->th_base + th_rel;
    s->th_m = p->motor_speed * s->t;
    s->ratio = twin_table_ratio(p, s->psi);
    s->a_t =  Rp * s->alpha;
    s->a_r = -Rp * s->omega * s->omega;

    /* --- 受け渡し（停留に入った最初のステップで 1 回だけ）----------- */
    if (!s->in_index && !s->transferred) {
        int out = s->i_out, in = s->i_in;
        s->transferred = 1;
        if (s->has_bottle[out]) {
            twin_event *e = push_event(s, TWIN_EV_DISCHARGE, out);
            double tilt = sqrt(s->phi_t[out] * s->phi_t[out] + s->phi_r[out] * s->phi_r[out]);
            if (e) {
                e->d[0] = s->V[out];
                e->d[1] = tilt;
                e->d[2] = s->phi_t[out];
                e->d[3] = s->phi_r[out];
                e->d[4] = s->dphi_t[out];
                e->d[5] = s->dphi_r[out];
                e->d[6] = holder_mass(s, out);
            }
            s->n_discharged++;
            s->out_volume += s->V[out];
            s->has_bottle[out] = 0;
            s->V[out] = 0.0;
            s->filled[out] = 0;
            s->phi_t[out] = s->phi_r[out] = 0.0;
            s->dphi_t[out] = s->dphi_r[out] = 0.0;
            s->dzr[out] = 0.0;
        }
        if (!s->has_bottle[in] && in != missing) {
            twin_event *e;
            s->has_bottle[in] = 1;
            s->V[in] = 0.0;
            s->filled[in] = 0;
            s->phi_t[in] = s->phi_r[in] = 0.0;
            s->dphi_t[in] = s->dphi_r[in] = 0.0;
            e = push_event(s, TWIN_EV_INFEED, in);
            /* 星車との当たりの大きさは params.json に無い。速度も剛性も置き場が
             * 無いので、ここでは発生時刻と空瓶の質量だけを渡す。 */
            if (e) e->d[0] = p->bottle_mass;
            s->n_infeed++;
        } else if (in == missing) {
            push_event(s, TWIN_EV_INFEED_MISSED, in);
            s->n_missed++;
        }
    }

    /* --- 充填（充填ステーションのホルダに 1 回だけ）----------------- */
    s->i_fill = twin_holder_at(p, p->fill_angle, s->th_t);
    s->filling = 0;
    {
        int fi = s->i_fill;
        if (!s->in_index && s->has_bottle[fi] && !s->filled[fi] && s->t >= s->valve_open_t) {
            if (!s->closing) {
                s->filling = 1;
                s->V[fi] = s->V[fi] + p->flow_rate * dt;
                if (s->V[fi] >= close_command_volume(p)) {
                    s->closing = 1;
                    s->close_t = s->t;
                }
            } else if (s->t < s->close_t + p->valve_close_delay + extra_close_delay) {
                s->filling = 1;
                s->V[fi] = s->V[fi] + p->flow_rate * dt;
            } else {
                s->filled[fi] = 1;
                if (drip_volume > 0.0) {
                    twin_event *e = push_event(s, TWIN_EV_VALVE_DRIP, fi);
                    if (e) e->d[0] = drip_volume;
                }
            }
        }
    }

    /* --- 各ホルダのスロッシング ------------------------------------- */
    for (i = 0; i < p->n_stations; i++) {
        double h, w1, L1, tilt, m, spilled, fl_t, fl_r, loc_t, loc_r, ang;
        double m_liq, m1, a1_t, a1_r;

        s->h[i] = s->V[i] / A;
        if (!s->has_bottle[i] || s->V[i] <= 0.0) {
            s->w1[i] = 0.0; s->L1[i] = 0.0;
            s->m0[i] = 0.0; s->m1[i] = 0.0;
            s->dzr[i] = 0.0;
            continue;
        }

        h = s->h[i];
        /* 充填で h が上がると w1 が変わる。毎ステップ計算し直す。 */
        w1 = twin_slosh_omega(p, h);
        L1 = pendulum_length(p, w1);
        s->w1[i] = w1;
        s->L1[i] = L1;

        step_slosh(s, i, s->a_t, s->a_r, s->omega, s->alpha, w1, dt);

        tilt = sqrt(s->phi_t[i] * s->phi_t[i] + s->phi_r[i] * s->phi_r[i]);
        if (tilt > s->max_tilt) s->max_tilt = tilt;
        if (s->in_index) { if (tilt > s->max_tilt_index) s->max_tilt_index = tilt; }
        else             { if (tilt > s->max_tilt_dwell) s->max_tilt_dwell = tilt; }
        m = tan(tilt);
        s->dzr[i] = m;
        if (m > s->max_dzr) s->max_dzr = m;
        if (m > TWIN_RANGE_DZR) { s->range_bad[i] = 1; s->range_warned = 1; }

        s->V[i] = apply_spill(p, s->V[i], tilt, &spilled);
        s->spill[i] += spilled;
        s->spill_total += spilled;
        s->h[i] = s->V[i] / A;

        /* 液がボトルに返す力（局所座標）を世界座標へ回して合成する。
         * liquid_force は反作用（テーブルが受ける側）を返すので、
         * 空瓶の慣性力も同じ向きに揃える（テーブルが受けるのは -m*a）。
         * w1 / L1 はこぼれる前の液深のもの（py/ref.py と同じ）。 */
        liquid_force(s, i, s->V[i], s->a_t, s->a_r, w1, L1, &fl_t, &fl_r);
        loc_t = fl_t - p->bottle_mass * s->a_t;
        loc_r = fl_r - p->bottle_mass * s->a_r;
        ang = station_world_angle(p, i, s->th_t);
        f_world_x += loc_r * cos(ang) - loc_t * sin(ang);
        f_world_y += loc_r * sin(ang) + loc_t * cos(ang);

        /* 揺れる分がテーブル軸に返すトルク（m0 の分は J_load 側にある）。
         * m1 はこぼれる前の液深 h から作る（py/ref.py と同じ）。 */
        m_liq = p->rho * s->V[i];
        m1 = m_liq * twin_slosh_mass_ratio(p, h);
        slosh_accel(s, i, w1, L1, &a1_t, &a1_r);
        torque_slosh += -Rp * m1 * a1_t;

        s->m1[i] = m1;
        s->m0[i] = m_liq - m1;
    }
    s->t_slosh = torque_slosh;
    s->f_tab_x = f_world_x;      /* 世界座標 x。テーブルが受ける側の符号 */
    s->f_tab_y = f_world_y;

    /* --- 軸のトルク -------------------------------------------------
     * T_table = J_load*al + (dJ_load/dt)*om + c_visc*om - T_slosh
     * **T_slosh は引く。** テーブルが受ける側の符号で定義してあるので、
     * 駆動側に要るトルクにするには符号を返す。準静的極限では
     * -T_slosh = +m1*Rp^2*al となって、揺動質量が慣性として素直に足される。 */
    s->j_load = rigid_load_inertia(s);
    s->dj_dt = s->filling ? load_inertia_rate(p, s->V[s->i_fill]) : 0.0;
    s->t_table = s->j_load * s->alpha + s->dj_dt * s->omega
               + p->table_viscous * s->omega - s->t_slosh;
    /* 効率の掛け方は流れの向きで逆になる（MODEL.md 4.7）。
     * カムが負荷を駆動している間は損失ぶん余計に要り、負荷がカムを回している間は
     * 損失ぶん減って伝わる。ratio は常に 0 以上なので、向きを決めるのは T_table*om。 */
    {
        double t_cam = s->t_table * s->ratio;
        t_cam = (s->t_table * s->omega >= 0.0) ? t_cam / p->cam_efficiency
                                               : t_cam * p->cam_efficiency;
        s->t_input = t_cam + p->input_drag_torque;
    }

    /* --- 垂直荷重（充填中はジェットの運動量も足す）------------------ */
    if (s->filling) { jet_station = s->i_fill; jet_F = jet_force(p); }
    bending_moment(s, s->th_t, jet_station, jet_F,
                   &s->m_bend_x, &s->m_bend_y, &s->m_bend);

    /* --- 診断 -------------------------------------------------------- */
    if (fabs(s->t_table) > s->max_t_table) s->max_t_table = fabs(s->t_table);
    /* T_in は符号付き（減速中は負になりうる）。走査は py/ref.py と同じ符号付き最大 */
    if (s->t_input > s->max_t_input) s->max_t_input = s->t_input;
    if (s->t_input < s->min_t_input) s->min_t_input = s->t_input;
    if (fabs(s->t_slosh) > s->max_t_slosh) s->max_t_slosh = fabs(s->t_slosh);
    if (s->m_bend > s->max_m_bend) s->max_m_bend = s->m_bend;
    {
        double fmag = hypot(s->f_tab_x, s->f_tab_y);
        double pw = fabs(s->t_table * s->omega);
        if (fmag > s->max_f_tab) s->max_f_tab = fmag;
        if (pw > s->max_power) s->max_power = pw;
    }

    /* --- 記録の判定（py/ref.py と同じ「t >= next_log」）-------------- */
    s->do_log = (s->t >= s->next_log);
    if (s->do_log) s->next_log = s->next_log + s->log_interval;

    /* --- 次のステップへ ---------------------------------------------- */
    s->step++;
    s->k++;
    if (s->k >= s->n_steps_cycle) {
        flush_sched(s, INFINITY);     /* 最後のステップより後ろの予定を出し切る */
        s->k = 0;
        s->cycle++;
        s->th_base = s->th_base + p->index_angle;
    }
}

int twin_should_log(const twin_state *s) { return s->do_log; }

/* ================================================================== */
/* 出力レコード（FORMAT.md 2 節）                                      */
/* ================================================================== */

int twin_record_floats(const twin_state *s)
{
    return TWIN_SCALARS_PER_REC + TWIN_FIELDS_PER_ST * s->p.n_stations;
}

int twin_record(const twin_state *s, double *buf, int cap)
{
    int n = twin_record_floats(s);
    int i, k = 0;
    if (cap < n) return 0;
    buf[k++] = s->t;
    buf[k++] = s->psi;
    buf[k++] = s->th_t;
    buf[k++] = s->omega;
    buf[k++] = s->alpha;
    buf[k++] = s->th_m;
    buf[k++] = s->j_load;
    buf[k++] = s->t_table;
    buf[k++] = s->t_input;
    buf[k++] = s->t_slosh;
    buf[k++] = s->m_bend;
    buf[k++] = s->m_bend_x;
    buf[k++] = s->m_bend_y;
    buf[k++] = s->f_tab_x;
    buf[k++] = s->f_tab_y;
    for (i = 0; i < s->p.n_stations; i++) {
        buf[k++] = s->V[i];
        buf[k++] = s->h[i];
        buf[k++] = s->phi_t[i];
        buf[k++] = s->phi_r[i];
        buf[k++] = s->spill[i];
    }
    return k;
}

/* 在荷フラグ（present）を 1 バイトずつ詰める。詰めた個数を返す。
 * 連続ログの末尾に置く u8 の並び（FORMAT.md 2.3）。 */
int twin_present(const twin_state *s, unsigned char *buf, int cap)
{
    int i;
    if (cap < s->p.n_stations) return 0;
    for (i = 0; i < s->p.n_stations; i++)
        buf[i] = (unsigned char)(s->has_bottle[i] ? 1 : 0);
    return s->p.n_stations;
}

size_t twin_state_size(void) { return sizeof(twin_state); }

/* ================================================================== */
/* イベントの取り出し                                                  */
/* ================================================================== */

int twin_event_count(const twin_state *s) { return s->n_ev; }

int twin_event_get(const twin_state *s, int k, double *out)
{
    int i;
    if (!out || k < 0 || k >= s->n_ev) return 1;
    out[0] = s->ev[k].t;
    out[1] = (double)s->ev[k].kind;
    out[2] = (double)s->ev[k].station;
    out[3] = s->ev[k].th_t;
    out[4] = s->ev[k].th_m;
    for (i = 0; i < TWIN_EV_DATA; i++) out[5 + i] = s->ev[k].d[i];
    return 0;
}
