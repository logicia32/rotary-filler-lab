/* main.c -- ロータリー充填機 仮想モデル / 単体実行 CLI（params.json rev.3）
 *
 * params.json を読んで時系列を回し、バイナリを吐く。
 * バイト単位の並びは core/FORMAT.md を参照（そちらが取り決めの正）。
 */

#include "twin.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HEADER_BYTES 160
#define MAGIC "RFILL004"

/* イベント列（サイドカー）。並びは FORMAT.md 5 節が正。 */
#define EV_MAGIC        "RFEVT002"
#define EV_HEADER_BYTES 64
#define EV_RECORD_BYTES 56

static void usage(const char *prog)
{
    fprintf(stderr,
"使い方: %s [オプション]\n"
"  --params <path>        params.json のパス（既定: ../params.json）\n"
"  --out <path>           出力ファイル（既定: out.bin、\"-\" で標準出力）\n"
"  --events <path>        イベント列の出力先（既定: <out>.events）\n"
"  --no-events            イベント列を書かない\n"
"  --duration <s>         計算時間（既定: タクト 1 サイクル分）\n"
"  --cycles <n>           サイクル数で指定する（--duration より優先）\n"
"  --tact <s>             タクトを上書き\n"
"  --index-time <s>       割出し時間を上書き\n"
"  --log-rate <hz>        ログ周波数を上書き\n"
"  --dt <s>               積分刻みを上書き\n"
"  --jet-fall <m>         ノズル出口から液面までの落差（既定 0）\n"
"  --f64                  レコードを f64 で書く（py/ref.py との突き合わせ用）\n"
"  --no-prime             空のテーブルから始める（既定は工程配置から作った定常状態）\n"
"  --quiet                サマリを出さない\n"
"  --help                 この表示\n"
"\n"
"故障モード（すべて既定は無効。params.json の faults が初期値）:\n"
"  --fault-missing            欠品（ボトルが載らない）を有効にする\n"
"  --missing-station <i>      欠けるホルダ番号\n"
"  --fault-valve              弁の閉じ遅れを有効にする\n"
"  --valve-extra-delay <s>    追加の閉じ遅れ\n"
"  --drip-mL <x>              液垂れ体積 [mL]\n"
"  --fault-cam                カムフォロワ摩耗を有効にする\n"
"  --cam-torque <Nm>          当たりの衝撃トルク\n"
"  --cam-clearance-deg <d>    すきま [deg]\n"
"  --fault-bearing            軸受外輪傷を有効にする\n"
"  --bearing-accel <a>        衝撃の振幅 [m/s^2]\n"
"  --bearing-freq <hz>        欠陥通過周波数 [Hz]\n"
"  --no-faults                params.json 側で有効でも全部無効にする\n", prog);
}

static int argnum(int argc, char **argv, int *i, double *out, const char *name)
{
    char *end = NULL;
    if (*i + 1 >= argc) { fprintf(stderr, "エラー: %s に値がない\n", name); return 1; }
    (*i)++;
    *out = strtod(argv[*i], &end);
    if (end == argv[*i] || (end && *end != '\0')) {
        fprintf(stderr, "エラー: %s の値が数値でない: %s\n", name, argv[*i]);
        return 1;
    }
    return 0;
}

static int is_little_endian(void)
{
    unsigned int x = 1u;
    return *(unsigned char *)&x == 1u;
}

static void put_u32(unsigned char *p, unsigned int v)
{
    p[0] = (unsigned char)(v & 0xff);
    p[1] = (unsigned char)((v >> 8) & 0xff);
    p[2] = (unsigned char)((v >> 16) & 0xff);
    p[3] = (unsigned char)((v >> 24) & 0xff);
}

static void put_i32(unsigned char *p, int v)
{
    put_u32(p, (unsigned int)v);
}

static void put_f32(unsigned char *p, double v)
{
    float f = (float)v;
    memcpy(p, &f, 4);
}

static void put_f64(unsigned char *p, double v)
{
    memcpy(p, &v, 8);
}

/* イベント列のファイル名を <out>.events に決める。out が "-" なら作らない。 */
static int default_events_path(const char *out_path, char *buf, size_t cap)
{
    size_t n = strlen(out_path);
    if (!strcmp(out_path, "-")) return 1;
    if (n + 8 >= cap) return 1;
    memcpy(buf, out_path, n);
    memcpy(buf + n, ".events", 8);      /* 終端 NUL 込み */
    return 0;
}

int main(int argc, char **argv)
{
    const char *params_path = "../params.json";
    const char *out_path    = "out.bin";
    const char *ev_path     = NULL;
    char        ev_buf[512];
    int         no_events = 0, prime = TWIN_PRIME_GEOM, use_f64 = 0, quiet = 0;
    FILE       *fev = NULL;
    unsigned int n_ev_written = 0u;

    double ov_tact = -1.0, ov_index = -1.0, ov_duration = -1.0, ov_cycles = -1.0;
    double ov_log = -1.0, ov_dt = -1.0, ov_jet = -1.0;
    double ov_missing_station = -1.0;
    double ov_valve_delay = -1.0, ov_drip = -1.0;
    double ov_cam_torque = -1.0, ov_cam_clear = -1.0;
    double ov_bear_accel = -1.0, ov_bear_freq = -1.0;
    int en_missing = 0, en_valve = 0, en_cam = 0, en_bear = 0, no_faults = 0;
    int i;

    twin_params  P;
    static twin_state S;
    char err[256];
    char warn[768];

    FILE *f = NULL;
    unsigned char hdr[HEADER_BYTES];
    double rec[TWIN_MAX_REC_FLOATS];
    float  rec32[TWIN_MAX_REC_FLOATS];
    unsigned char flags[TWIN_MAX_STATIONS + 8];
    int    nfloat, nflag, flag_bytes;
    long   nstep, k;
    unsigned int nrec = 0u;
    unsigned int fault_flags = 0u;
    unsigned int elem_bytes;
    double duration;
    int n_full0 = 0, n_empty0 = 0, n_free0 = 0;   /* 初期状態の内訳（サマリ用） */

    for (i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--params") && i + 1 < argc) params_path = argv[++i];
        else if (!strcmp(argv[i], "--out")    && i + 1 < argc) out_path    = argv[++i];
        else if (!strcmp(argv[i], "--events") && i + 1 < argc) ev_path     = argv[++i];
        else if (!strcmp(argv[i], "--no-events")) { no_events = 1; }
        else if (!strcmp(argv[i], "--no-prime"))  { prime = TWIN_PRIME_EMPTY; }
        else if (!strcmp(argv[i], "--f64"))       { use_f64 = 1; }
        else if (!strcmp(argv[i], "--quiet"))     { quiet = 1; }
        else if (!strcmp(argv[i], "--tact"))       { if (argnum(argc, argv, &i, &ov_tact, "--tact")) return 2; }
        else if (!strcmp(argv[i], "--index-time")) { if (argnum(argc, argv, &i, &ov_index, "--index-time")) return 2; }
        else if (!strcmp(argv[i], "--duration"))   { if (argnum(argc, argv, &i, &ov_duration, "--duration")) return 2; }
        else if (!strcmp(argv[i], "--cycles"))     { if (argnum(argc, argv, &i, &ov_cycles, "--cycles")) return 2; }
        else if (!strcmp(argv[i], "--log-rate"))   { if (argnum(argc, argv, &i, &ov_log, "--log-rate")) return 2; }
        else if (!strcmp(argv[i], "--dt"))         { if (argnum(argc, argv, &i, &ov_dt, "--dt")) return 2; }
        else if (!strcmp(argv[i], "--jet-fall"))   { if (argnum(argc, argv, &i, &ov_jet, "--jet-fall")) return 2; }
        else if (!strcmp(argv[i], "--fault-missing")) { en_missing = 1; }
        else if (!strcmp(argv[i], "--fault-valve"))   { en_valve = 1; }
        else if (!strcmp(argv[i], "--fault-cam"))     { en_cam = 1; }
        else if (!strcmp(argv[i], "--fault-bearing")) { en_bear = 1; }
        else if (!strcmp(argv[i], "--no-faults"))     { no_faults = 1; }
        else if (!strcmp(argv[i], "--missing-station")) { if (argnum(argc, argv, &i, &ov_missing_station, "--missing-station")) return 2; en_missing = 1; }
        else if (!strcmp(argv[i], "--valve-extra-delay")) { if (argnum(argc, argv, &i, &ov_valve_delay, "--valve-extra-delay")) return 2; en_valve = 1; }
        else if (!strcmp(argv[i], "--drip-mL"))       { if (argnum(argc, argv, &i, &ov_drip, "--drip-mL")) return 2; en_valve = 1; }
        else if (!strcmp(argv[i], "--cam-torque"))    { if (argnum(argc, argv, &i, &ov_cam_torque, "--cam-torque")) return 2; en_cam = 1; }
        else if (!strcmp(argv[i], "--cam-clearance-deg")) { if (argnum(argc, argv, &i, &ov_cam_clear, "--cam-clearance-deg")) return 2; en_cam = 1; }
        else if (!strcmp(argv[i], "--bearing-accel")) { if (argnum(argc, argv, &i, &ov_bear_accel, "--bearing-accel")) return 2; en_bear = 1; }
        else if (!strcmp(argv[i], "--bearing-freq"))  { if (argnum(argc, argv, &i, &ov_bear_freq, "--bearing-freq")) return 2; en_bear = 1; }
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "エラー: 不明な引数: %s\n", argv[i]); usage(argv[0]); return 2; }
    }

    if (!is_little_endian()) {
        fprintf(stderr, "エラー: 出力フォーマットはリトルエンディアン前提です\n");
        return 1;
    }

    if (twin_params_load(&P, params_path, err, sizeof(err))) {
        fprintf(stderr, "エラー: %s\n", err);
        return 1;
    }

    /* コマンドラインの上書き */
    if (ov_tact  > 0.0) P.tact       = ov_tact;
    if (ov_index > 0.0) P.index_time = ov_index;
    if (ov_log   > 0.0) P.log_rate   = ov_log;
    if (ov_dt    > 0.0) P.dt         = ov_dt;
    if (ov_jet  >= 0.0) P.jet_fall_height = ov_jet;
    P.dwell = P.tact - P.index_time;
    if (P.index_time > P.tact) {
        fprintf(stderr, "エラー: index_time (%.4g s) が tact (%.4g s) を超えている\n",
                P.index_time, P.tact);
        return 1;
    }
    duration = (ov_duration > 0.0) ? ov_duration : P.tact;
    if (ov_cycles > 0.0) duration = ov_cycles * P.tact;

    /* 故障モードの有効化。CLI で指定したものは params.json より優先する。 */
    if (en_missing) P.flt.missing_enabled = 1;
    if (en_valve)   P.flt.valve_enabled   = 1;
    if (en_cam)     P.flt.cam_enabled     = 1;
    if (en_bear)    P.flt.bear_enabled    = 1;
    if (no_faults) {
        P.flt.missing_enabled = P.flt.valve_enabled = 0;
        P.flt.cam_enabled = P.flt.bear_enabled = 0;
    }
    if (ov_missing_station >= 0.0) P.flt.missing_station = (int)(ov_missing_station + 0.5);
    if (ov_valve_delay >= 0.0) P.flt.valve_extra_delay = ov_valve_delay;
    if (ov_drip        >= 0.0) P.flt.valve_drip_volume = ov_drip * 1e-6;
    if (ov_cam_torque  >= 0.0) P.flt.cam_impact_torque = ov_cam_torque;
    if (ov_cam_clear   >= 0.0) P.flt.cam_clearance     = ov_cam_clear * (M_PI / 180.0);
    if (ov_bear_accel  >= 0.0) P.flt.bear_accel        = ov_bear_accel;
    if (ov_bear_freq    > 0.0) P.flt.bear_defect_freq  = ov_bear_freq;

    if (P.flt.missing_station < 0 || P.flt.missing_station >= P.n_stations) {
        fprintf(stderr, "エラー: --missing-station が範囲外: %d（0..%d）\n",
                P.flt.missing_station, P.n_stations - 1);
        return 1;
    }
    if (P.flt.bear_enabled && !(P.flt.bear_defect_freq > 0.0)) {
        fprintf(stderr, "エラー: 軸受の欠陥通過周波数が 0 では衝撃が出ない"
                        "（--bearing-freq で指定）\n");
        return 1;
    }
    if (P.flt.bear_enabled && P.flt.bear_defect_freq * P.tact > TWIN_MAX_SCHED - 4) {
        fprintf(stderr, "エラー: 1 タクトの軸受衝撃が %d 件を超える（%.0f 件）。"
                        " TWIN_MAX_SCHED を上げること\n",
                TWIN_MAX_SCHED - 4, P.flt.bear_defect_freq * P.tact);
        return 1;
    }

    if (P.flt.missing_enabled) fault_flags |= 1u;
    if (P.flt.valve_enabled)   fault_flags |= 2u;
    if (P.flt.cam_enabled)     fault_flags |= 4u;
    if (P.flt.bear_enabled)    fault_flags |= 8u;

    twin_init(&S, &P, prime);
    for (i = 0; i < P.n_stations; i++) {
        if (!S.has_bottle[i]) n_free0++;
        else if (S.V[i] > 0.0) n_full0++;
        else n_empty0++;
    }

    if (twin_check_dt(&S, warn, sizeof(warn)))
        fputs(warn, stderr);

    /* 充填が停留時間に収まるか（params.json の意図と実際のずれを見る） */
    {
        double t_fill = P.start_delay + P.valve_open_delay
                      + P.target_volume / P.flow_rate + P.valve_close_delay;
        double h_fill = P.target_volume / (M_PI * P.bottle_R * P.bottle_R);
        if (t_fill > P.dwell)
            fprintf(stderr,
                "注意: 充填に必要な時間 %.4f s が停留時間 %.4f s を超える"
                "（弁は停留を跨いで開いたままになる）\n", t_fill, P.dwell);
        if (h_fill > P.body_height)
            fprintf(stderr,
                "注意: 充填量 %.1f mL は胴部高さ %.1f mm に入らない（液深 %.1f mm）\n",
                P.target_volume * 1e6, P.body_height * 1e3, h_fill * 1e3);
    }

    nfloat = twin_record_floats(&S);
    elem_bytes = use_f64 ? 8u : 4u;
    /* 在荷フラグは u8。レコード長が elem_bytes の倍数になるように詰め物をする
     * （読む側が浮動小数の並びを揃った境界で切れるように）。 */
    nflag = P.n_stations;
    flag_bytes = (int)(((unsigned)nflag + elem_bytes - 1u) / elem_bytes * elem_bytes);
    nstep = (long)floor(duration / P.dt + 0.5);

    if (!strcmp(out_path, "-")) f = stdout;
    else f = fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "エラー: 出力を開けない: %s\n", out_path); return 1; }

    /* ---- ヘッダ（FORMAT.md 1 節） ---- */
    memset(hdr, 0, sizeof(hdr));
    memcpy(hdr + 0, MAGIC, 8);
    put_u32(hdr +  8, (unsigned int)HEADER_BYTES);
    put_u32(hdr + 12, elem_bytes * (unsigned int)nfloat + (unsigned int)flag_bytes);
    put_u32(hdr + 16, (unsigned int)P.n_stations);
    put_u32(hdr + 20, (unsigned int)TWIN_SCALARS_PER_REC);
    put_u32(hdr + 24, (unsigned int)TWIN_FIELDS_PER_ST);
    put_u32(hdr + 28, elem_bytes);
    put_u32(hdr + 32, 0u);                              /* レコード数（最後に書き戻す） */
    put_u32(hdr + 36, fault_flags);
    put_f64(hdr + 40, P.dt);
    put_f64(hdr + 48, 1.0 / P.log_rate);
    put_f32(hdr + 56, P.Rp);
    put_f32(hdr + 60, P.bottle_R);
    put_f32(hdr + 64, P.body_height);
    put_f32(hdr + 68, 0.0);                             /* 最大傾き [rad]、書き戻し */
    put_f32(hdr + 72, 0.0);                             /* 最大 dz/R、書き戻し */
    put_u32(hdr + 76, 0u);                              /* 適用範囲逸脱、書き戻し */
    put_f32(hdr + 80, TWIN_RANGE_DZR);
    put_f32(hdr + 84, P.tact);
    put_f32(hdr + 88, P.index_time);
    put_f32(hdr + 92, P.dwell);
    put_f32(hdr + 96, P.index_angle);
    put_f32(hdr +100, 2.0 * M_PI / P.tact);
    put_f32(hdr +104, P.motor_speed);
    put_f32(hdr +108, P.gear_ratio);
    put_f32(hdr +112, P.target_volume);
    put_f32(hdr +116, P.flow_rate);
    {
        double h_full = P.target_volume / (M_PI * P.bottle_R * P.bottle_R);
        put_f32(hdr +120, twin_slosh_omega(&P, h_full));
    }
    put_f32(hdr +124, P.infeed_angle);
    put_f32(hdr +128, P.fill_angle);
    put_f32(hdr +132, P.discharge_angle);
    put_f32(hdr +136, P.cam_efficiency);
    put_f32(hdr +140, P.input_drag_torque);
    put_f32(hdr +144, P.table_viscous);
    put_f32(hdr +148, P.flt.bear_enabled ? P.flt.bear_defect_freq : 0.0);
    put_i32(hdr +152, P.flt.missing_enabled ? P.flt.missing_station : -1);
    put_u32(hdr +156, (unsigned int)flag_bytes);   /* 末尾の在荷フラグ（詰め物込み）*/

    if (fwrite(hdr, 1, sizeof(hdr), f) != sizeof(hdr)) {
        fprintf(stderr, "エラー: ヘッダの書き込みに失敗\n");
        if (f != stdout) fclose(f);
        return 1;
    }

    /* ---- イベント列（サイドカー。FORMAT.md 5 節） ---- */
    if (!no_events) {
        if (!ev_path && !default_events_path(out_path, ev_buf, sizeof(ev_buf)))
            ev_path = ev_buf;
        if (ev_path) {
            fev = fopen(ev_path, "wb");
            if (!fev) {
                fprintf(stderr, "エラー: イベント列の出力を開けない: %s\n", ev_path);
                if (f != stdout) fclose(f);
                return 1;
            }
            memset(hdr, 0, EV_HEADER_BYTES);
            memcpy(hdr + 0, EV_MAGIC, 8);
            put_u32(hdr +  8, (unsigned int)EV_HEADER_BYTES);
            put_u32(hdr + 12, (unsigned int)EV_RECORD_BYTES);
            put_u32(hdr + 16, 0u);                      /* 件数、最後に書き戻す */
            put_u32(hdr + 20, fault_flags);
            /* CLI で上書きされた「実効値」を書く。センサ層はこれを見て波形を合成する。 */
            put_f32(hdr + 24, P.flt.bear_enabled ? P.flt.bear_ring_freq    : 0.0);
            put_f32(hdr + 28, P.flt.bear_enabled ? P.flt.bear_ring_damping : 0.0);
            put_f32(hdr + 32, P.flt.bear_enabled ? P.flt.bear_accel        : 0.0);
            put_f32(hdr + 36, P.flt.bear_enabled ? P.flt.bear_defect_freq  : 0.0);
            put_f32(hdr + 40, P.flt.cam_enabled  ? P.flt.cam_impact_torque : 0.0);
            put_f32(hdr + 44, P.flt.cam_enabled  ? P.flt.cam_clearance     : 0.0);
            put_f64(hdr + 48, P.dt);
            put_f32(hdr + 56, duration);
            put_u32(hdr + 60, 0u);
            if (fwrite(hdr, 1, EV_HEADER_BYTES, fev) != EV_HEADER_BYTES) {
                fprintf(stderr, "エラー: イベント列のヘッダの書き込みに失敗\n");
                if (f != stdout) fclose(f);
                fclose(fev);
                return 1;
            }
        }
    }

    /* ---- 本計算 ----
     * twin_step は 1 ステップぶんの物理を片付けてから戻る。戻った直後の状態が
     * そのステップの時刻に対応するので、記録は step のあと（py/ref.py と同じ）。 */
    for (k = 0; k < nstep; k++) {
        int e, ne;
        twin_step(&S);
        ne = twin_event_count(&S);
        for (e = 0; e < ne && fev; e++) {
            unsigned char erec[EV_RECORD_BYTES];
            double v[5 + TWIN_EV_DATA];
            int d;
            twin_event_get(&S, e, v);
            memset(erec, 0, sizeof(erec));
            put_f64(erec +  0, v[0]);                    /* 時刻 [s] */
            put_u32(erec +  8, (unsigned int)v[1]);      /* 種別 */
            put_i32(erec + 12, (int)v[2]);               /* ホルダ番号 */
            put_f32(erec + 16, v[3]);                    /* テーブル角 */
            put_f32(erec + 20, v[4]);                    /* モータ軸角 */
            for (d = 0; d < TWIN_EV_DATA; d++)
                put_f32(erec + 24 + 4 * d, v[5 + d]);
            if (fwrite(erec, 1, sizeof(erec), fev) != sizeof(erec)) {
                fprintf(stderr, "エラー: イベントの書き込みに失敗\n");
                if (f != stdout) fclose(f);
                fclose(fev);
                return 1;
            }
            n_ev_written++;
        }
        if (twin_should_log(&S)) {
            size_t want = (size_t)nfloat;
            twin_record(&S, rec, TWIN_MAX_REC_FLOATS);
            if (use_f64) {
                if (fwrite(rec, sizeof(double), want, f) != want) goto write_fail;
            } else {
                int j;
                for (j = 0; j < nfloat; j++) rec32[j] = (float)rec[j];
                if (fwrite(rec32, sizeof(float), want, f) != want) goto write_fail;
            }
            memset(flags, 0, sizeof(flags));
            twin_present(&S, flags, (int)sizeof(flags));
            if (fwrite(flags, 1, (size_t)flag_bytes, f) != (size_t)flag_bytes)
                goto write_fail;
            nrec++;
        }
    }

    if (fev) {
        unsigned char b4[4];
        if (fseek(fev, 16, SEEK_SET) == 0) { put_u32(b4, n_ev_written); fwrite(b4, 1, 4, fev); }
        fclose(fev);
        fev = NULL;
    }

    /* レコード数・最大傾き・適用範囲を書き戻す（通常ファイルのときだけ） */
    if (f != stdout) {
        unsigned char b4[4];
        if (fseek(f, 32, SEEK_SET) == 0) { put_u32(b4, nrec); fwrite(b4, 1, 4, f); }
        if (fseek(f, 68, SEEK_SET) == 0) { put_f32(b4, S.max_tilt); fwrite(b4, 1, 4, f); }
        if (fseek(f, 72, SEEK_SET) == 0) { put_f32(b4, S.max_dzr); fwrite(b4, 1, 4, f); }
        if (fseek(f, 76, SEEK_SET) == 0) { put_u32(b4, (unsigned int)(S.range_warned ? 1 : 0)); fwrite(b4, 1, 4, f); }
        fclose(f);
    } else {
        fflush(stdout);
    }

    /* ---- サマリ（標準エラーへ） ---- */
    if (!quiet) {
        fprintf(stderr, "----- 計算条件 -----\n");
        fprintf(stderr, "params      : %s\n", params_path);
        fprintf(stderr, "出力        : %s (%u レコード x %u byte = %.2f MB, %s + 在荷 u8 x%d)\n",
                out_path, nrec, elem_bytes * (unsigned int)nfloat + (unsigned int)flag_bytes,
                (double)nrec * (elem_bytes * nfloat + flag_bytes) / 1048576.0,
                use_f64 ? "f64" : "f32", nflag);
        if (ev_path)
            fprintf(stderr, "イベント列  : %s (%u 件)\n", ev_path, n_ev_written);
        fprintf(stderr, "初期状態    : %s（満量 %d + 空瓶 %d + 空ホルダ %d）\n",
                prime == TWIN_PRIME_GEOM ? "工程配置から" : "空のテーブル",
                n_full0, n_empty0, n_free0);
        fprintf(stderr, "工程        : 供給 %.0f deg / 充填 %.0f deg / 排出 %.0f deg\n",
                P.infeed_angle * 180.0 / M_PI, P.fill_angle * 180.0 / M_PI,
                P.discharge_angle * 180.0 / M_PI);
        fprintf(stderr, "tact/index  : %.4f s / %.4f s (停留 %.4f s)\n",
                P.tact, P.index_time, P.dwell);
        fprintf(stderr, "入力軸      : %.4f rad/s (%.1f rpm) / モータ %.2f rad/s (%.0f rpm)\n",
                2.0 * M_PI / P.tact, 60.0 / P.tact,
                P.motor_speed, P.motor_speed * 60.0 / (2.0 * M_PI));
        fprintf(stderr, "dt / log    : %.3g s / %.4g Hz\n", P.dt, P.log_rate);
        fprintf(stderr, "摩擦(仮置き): 効率 %.3f / 引きずり %.3f Nm / 粘性 %.3f Nm s/rad\n",
                P.cam_efficiency, P.input_drag_torque, P.table_viscous);
        fprintf(stderr, "故障モード  : 欠品=%s 弁閉じ遅れ=%s カムフォロワ=%s 軸受=%s\n",
                P.flt.missing_enabled ? "有効" : "無効",
                P.flt.valve_enabled   ? "有効" : "無効",
                P.flt.cam_enabled     ? "有効" : "無効",
                P.flt.bear_enabled    ? "有効" : "無効");
        if (P.flt.missing_enabled)
            fprintf(stderr, "  欠品: st%d にボトルが載らない\n", P.flt.missing_station);
        if (P.flt.valve_enabled)
            fprintf(stderr, "  弁: 追加遅れ %.4g s / 液垂れ %.4g mL\n",
                    P.flt.valve_extra_delay, P.flt.valve_drip_volume * 1e6);
        if (P.flt.cam_enabled)
            fprintf(stderr, "  カムフォロワ: すきま %.4g deg / 衝撃 %.4g Nm\n",
                    P.flt.cam_clearance * 180.0 / M_PI, P.flt.cam_impact_torque);
        if (P.flt.bear_enabled)
            fprintf(stderr, "  軸受: 欠陥 %.4g Hz / 振幅 %.4g m/s^2 / リンギング %.4g Hz (zeta %.4g)\n",
                    P.flt.bear_defect_freq, P.flt.bear_accel,
                    P.flt.bear_ring_freq, P.flt.bear_ring_damping);

        fprintf(stderr, "----- 結果 -----\n");
        fprintf(stderr, "t_end       : %.6f s (%d サイクル)\n", S.t, S.cycle);
        fprintf(stderr, "テーブル角  : %.6f rad = %.2f deg\n", S.th_t, S.th_t * 180.0 / M_PI);
        fprintf(stderr, "負荷慣性    : %.6f kg m^2 (液は m0 のみ)\n", S.j_load);
        fprintf(stderr, "テーブル軸トルク max : %.6f Nm\n", S.max_t_table);
        fprintf(stderr, "入力軸トルク         : %.6f 〜 %.6f Nm (引きずり %.3f。減速中は負)\n",
                S.min_t_input, S.max_t_input, P.input_drag_torque);
        fprintf(stderr, "ピーク機械出力       : %.6f W\n", S.max_power);
        fprintf(stderr, "反力トルク max       : %.6f Nm\n", S.max_t_slosh);
        fprintf(stderr, "水平合力 max         : %.6f N\n", S.max_f_tab);
        fprintf(stderr, "曲げモーメント max   : %.6f Nm\n", S.max_m_bend);
        fprintf(stderr, "合成傾き max : %.6f rad = %.3f mrad（割出し中 %.3f / 停留中 %.3f mrad）\n",
                S.max_tilt, S.max_tilt * 1e3, S.max_tilt_index * 1e3, S.max_tilt_dwell * 1e3);
        fprintf(stderr, "dz/R max    : %.6f （適用範囲 %.2f）%s\n",
                S.max_dzr, TWIN_RANGE_DZR,
                S.range_warned ? "  ← 逸脱あり。1 次モード近似の外" : "  適用範囲内");
        fprintf(stderr, "受け渡し    : 排出 %ld 本 (%.1f mL) / 供給 %ld 本 / 欠品 %ld 回\n",
                S.n_discharged, S.out_volume * 1e6, S.n_infeed, S.n_missed);
        fprintf(stderr, "こぼれ      : %.6g mL\n", S.spill_total * 1e6);
        if (S.n_ev_lost)
            fprintf(stderr, "注意: イベントを %ld 件取りこぼした（1 ステップ %d 件の上限）\n",
                    S.n_ev_lost, TWIN_MAX_EVENTS);
        for (i = 0; i < P.n_stations; i++) {
            fprintf(stderr,
                "st%-2d %s V=%8.3f mL  h=%7.2f mm  w1=%7.3f rad/s (%6.3f Hz)  "
                "m1=%6.4f kg  spill=%.5f mL%s\n",
                i, S.has_bottle[i] ? "瓶有" : "空  ", S.V[i] * 1e6, S.h[i] * 1e3, S.w1[i],
                S.w1[i] / (2.0 * M_PI), S.m1[i], S.spill[i] * 1e6,
                S.range_bad[i] ? "  ← 適用範囲逸脱" : "");
        }
    }
    return 0;

write_fail:
    fprintf(stderr, "エラー: レコードの書き込みに失敗\n");
    if (f != stdout) fclose(f);
    if (fev) fclose(fev);
    return 1;
}
