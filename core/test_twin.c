/* test_twin.c -- C 物理コアの単体試験
 *
 * 期待値は `py/ref.py` の実測値（MODEL.md の諸元表と同じ数）。
 * ここで見るのは「C 単体で、諸元から出る量が合っているか」まで。
 * 時系列そのものの突き合わせは core/compare_ref.py（同じ params で
 * py/ref.py の simulate() を回して 1 レコードずつ比べる）が担当する。
 *
 * 期待値は params.json rev.3 に対するもので、params.json を動かすと当然ずれる。
 * ずれたときに直すのは**期待値ではなく params.json か実装**のどちらか。
 * 「合わせにいく」ためにここの数字を書き換えないこと。
 *
 *     ./test_twin [--params <path>]
 */

#include "twin.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int n_check = 0;
static int n_fail  = 0;

/* 相対差で見る。期待値が 0 に近いときは絶対差に落とす。 */
static void check(const char *name, double got, double want, double tol,
                  const char *unit)
{
    double err = fabs(got - want);
    double rel = (fabs(want) > 1e-12) ? err / fabs(want) : err;
    int ok = (rel <= tol) && !isnan(got);
    n_check++;
    if (!ok) n_fail++;
    printf("  %s %-38s %14.6g %-10s (期待 %.6g, 差 %.2e, 許容 %.0e)\n",
           ok ? "OK " : "NG ", name, got, unit, want, rel, tol);
}

static void check_int(const char *name, long got, long want)
{
    int ok = (got == want);
    n_check++;
    if (!ok) n_fail++;
    printf("  %s %-38s %14ld            (期待 %ld)\n",
           ok ? "OK " : "NG ", name, got, want);
}

/* ------------------------------------------------------------------ */
/* 諸元から一意に決まる量                                              */
/* ------------------------------------------------------------------ */

static double table_omega_max(const twin_params *p)
{
    const double Cv = (4.0 * M_PI * M_PI / (M_PI + 4.0)) / M_PI;
    return Cv * p->index_angle / p->index_time;
}

static double table_alpha_max(const twin_params *p)
{
    const double Ca = 4.0 * M_PI * M_PI / (M_PI + 4.0);
    return Ca * p->index_angle / (p->index_time * p->index_time);
}

/* 工程配置から作った定常状態の、液を全量剛体として数えた J_load。
 * 諸元表に載るのはこちら。実装（twin.c）が使うのは m0 だけの値で、そちらが少し小さい。
 * ホルダの並びは twin_init(TWIN_PRIME_GEOM) が作るものをそのまま使う
 * （工程角を動かすと並びが変わるので、数を決め打ちにしない）。 */
static double steady_rigid_all(const twin_params *p)
{
    static twin_state S;
    double m = 0.0;
    int i;
    twin_init(&S, p, TWIN_PRIME_GEOM);
    for (i = 0; i < p->n_stations; i++)
        if (S.has_bottle[i]) m += p->bottle_mass + p->rho * S.V[i];
    return p->bare_inertia + m * p->Rp * p->Rp;
}

/* 供給から排出までの割出し回数（滞留）。工程角から一意に決まる。 */
static int residence_indexes(const twin_params *p)
{
    double pitch = 2.0 * M_PI / (double)p->n_stations;
    int k = (int)nearbyint((p->discharge_angle - p->infeed_angle) / pitch);
    return ((k % p->n_stations) + p->n_stations) % p->n_stations;
}

/* 割出し 1 回の中での入力軸トルクの最大 [Nm] と、そのときの psi [rad]。
 * トルクの最大（psi = 22.5deg）と速度の最大（psi = 90deg）は同時に起きない。
 * 2 つの最大値を掛けると 8 割ほど過大になるので走査する。 */
static double input_torque_peak(const twin_params *p, double j, int viscous,
                                int drag, double *psi_at)
{
    const int n = 20000;
    double best = 0.0, best_psi = 0.0;
    int k;
    for (k = 0; k <= n; k++) {
        double psi = p->index_angle_input * (double)k / (double)n;
        double th, om, al, t_tab, t_in;
        twin_table_motion(p, psi, &th, &om, &al);
        t_tab = j * al + (viscous ? p->table_viscous * om : 0.0);
        t_in = fabs(t_tab * twin_table_ratio(p, psi)) / p->cam_efficiency;
        if (drag) t_in += p->input_drag_torque;
        if (t_in > best) { best = t_in; best_psi = psi; }
    }
    if (psi_at) *psi_at = best_psi;
    return best;
}

static double peak_power(const twin_params *p, double j, int viscous)
{
    const int n = 20000;
    double best = 0.0;
    int k;
    for (k = 0; k <= n; k++) {
        double psi = p->index_angle_input * (double)k / (double)n;
        double th, om, al, t_tab;
        twin_table_motion(p, psi, &th, &om, &al);
        t_tab = j * al + (viscous ? p->table_viscous * om : 0.0);
        if (fabs(t_tab * om) > best) best = fabs(t_tab * om);
    }
    return best;
}

/* 静止した液面に割出しを 1 回かけたときの傾き。1 本ぶんだけを見る。
 * py/ref.py の index_slosh_response と同じ条件（満量固定・dt = 1e-4・1 タクト）。
 * ホルダ 0 だけに満量のボトルを置く。工程はそのまま回るが、ホルダ 0 は
 * 充填にも受け渡しにも当たらないので、1 本の自由な応答がそのまま出る。 */
static void index_slosh_response(const twin_params *base, int no_coriolis,
                                 double *peak_index, double *peak_dwell)
{
    static twin_state S;
    twin_params p = *base;
    long k, nstep;
    double pi_max = 0.0, pd_max = 0.0;

    p.dt = 1.0e-4;
    p.no_coriolis = no_coriolis;
    twin_init(&S, &p, 0);
    S.has_bottle[0] = 1;
    S.V[0] = p.target_volume;
    S.filled[0] = 1;
    twin_refresh(&S);

    nstep = (long)floor(p.tact / p.dt + 0.5);
    for (k = 0; k < nstep; k++) {
        double tilt;
        twin_step(&S);
        tilt = hypot(S.phi_t[0], S.phi_r[0]);
        if (S.in_index) { if (tilt > pi_max) pi_max = tilt; }
        else            { if (tilt > pd_max) pd_max = tilt; }
    }
    *peak_index = pi_max;
    *peak_dwell = pd_max;
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    const char *params_path = "../params.json";
    twin_params P;
    static twin_state S;
    char err[256];
    int i;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--params") && i + 1 < argc) params_path = argv[++i];
        else { fprintf(stderr, "使い方: %s [--params <path>]\n", argv[0]); return 2; }
    }

    if (twin_params_load(&P, params_path, err, sizeof(err))) {
        fprintf(stderr, "エラー: %s\n", err);
        return 1;
    }

    printf("C 物理コアの単体試験（params: %s）\n", params_path);

    /* -------------------------------------------------------------- */
    printf("\n--- 1. カム曲線（変形正弦） ---\n");
    {
        const double Ca = 4.0 * M_PI * M_PI / (M_PI + 4.0);
        double s, v, a, s1, v1, a1, smax = 0.0, vmax = 0.0, amax = 0.0;
        double s_at_amax = 0.0;
        int k;
        for (k = 0; k <= 200000; k++) {
            double x = (double)k / 200000.0;
            twin_modified_sine(x, &s, &v, &a);
            if (v > vmax) vmax = v;
            if (a > amax) { amax = a; s_at_amax = x; }
            if (s > smax) smax = s;
        }
        check("Ca（無次元最大加速度）", amax, 5.527957, 1e-5, "[-]");
        check("Cv（無次元最大速度）",   vmax, 1.759603, 1e-5, "[-]");
        check("Ca = 4pi^2/(pi+4)",      Ca,   5.527957, 1e-6, "[-]");
        check("Ca が出る位置 x",        s_at_amax, 0.125, 1e-4, "[-]");
        twin_modified_sine(1.0, &s, &v, &a);
        check("S(1) = 1", s, 1.0, 1e-12, "[-]");
        check("変位の最大 = 1", smax, 1.0, 1e-9, "[-]");
        /* 継ぎ目の連続性（ここが飛ぶと停留とのつなぎが壊れる） */
        twin_modified_sine(0.125 - 1e-9, &s, &v, &a);
        twin_modified_sine(0.125 + 1e-9, &s1, &v1, &a1);
        check("x=1/8 の速度の跳び", fabs(v - v1), 0.0, 1e-7, "[-]");
        check("x=1/8 の加速度の跳び", fabs(a - a1), 0.0, 1e-6, "[-]");
        twin_modified_sine(0.875 - 1e-9, &s, &v, &a);
        twin_modified_sine(0.875 + 1e-9, &s1, &v1, &a1);
        check("x=7/8 の速度の跳び", fabs(v - v1), 0.0, 1e-7, "[-]");
        check("x=7/8 の加速度の跳び", fabs(a - a1), 0.0, 1e-6, "[-]");
        check("params.json の curve_Ca との差", amax, 5.528, 1e-4, "[-]");
    }

    /* -------------------------------------------------------------- */
    printf("\n--- 2. 割出しの運動学 ---\n");
    {
        double om_max = table_omega_max(&P);
        double al_max = table_alpha_max(&P);
        double th, om, al, best_om = 0.0, best_al = 0.0;
        int k;
        for (k = 0; k <= 200000; k++) {
            double psi = P.index_angle_input * (double)k / 200000.0;
            twin_table_motion(&P, psi, &th, &om, &al);
            if (om > best_om) best_om = om;
            if (al > best_al) best_al = al;
        }
        check("最大角速度（走査）",   best_om, 0.92133, 1e-4, "rad/s");
        check("最大角加速度（走査）", best_al, 1.92962, 1e-4, "rad/s^2");
        check("最大角速度（式）",     om_max,  0.92133, 1e-4, "rad/s");
        check("最大角加速度（式）",   al_max,  1.92962, 1e-4, "rad/s^2");
        check("最大接線加速度",       P.Rp * al_max, 0.43416, 1e-4, "m/s^2");
        check("最大接線加速度 [g]",   P.Rp * al_max / P.g, 0.04427, 1e-3, "[g]");
        /* 停留は厳密に止まる（カムの形がそのまま停留を作る） */
        twin_table_motion(&P, P.index_angle_input + 0.1, &th, &om, &al);
        check("停留中の角速度", om, 0.0, 1e-15, "rad/s");
        check("停留中の角加速度", al, 0.0, 1e-15, "rad/s^2");
        check("停留中のテーブル角", th, P.index_angle, 1e-12, "rad");
        check("変速比の最大", twin_table_ratio(&P, P.index_angle_input / 2.0),
              0.43990, 1e-4, "[-]");
    }

    /* -------------------------------------------------------------- */
    printf("\n--- 3. 慣性 ---\n");
    {
        double r = P.plate_diameter / 2.0;
        double m_geo = M_PI * r * r * P.plate_thickness * P.plate_density;
        double j_geo = 0.5 * m_geo * r * r;
        check("板の質量（幾何から）", m_geo, 15.5665, 1e-4, "kg");
        check("板の極慣性 (1/2)mr^2", j_geo, 0.61021, 1e-4, "kg m^2");
        check("json の bare_inertia との差", P.bare_inertia, j_geo, 1e-4, "kg m^2");
        check("J_load（液を全量剛体）", steady_rigid_all(&P), 0.719044, 1e-4, "kg m^2");
        check("テーブル軸ピークトルク",
              steady_rigid_all(&P) * table_alpha_max(&P), 1.38748, 1e-4, "Nm");
        /* 実装が使うのは m0 だけの慣性。定常状態を作って確かめる。 */
        twin_init(&S, &P, TWIN_PRIME_GEOM);
        check("J_load（m0 のみ・実装）", S.j_load, 0.706637, 1e-4, "kg m^2");
        {
            int n_full = 0, n_empty = 0, n_free = 0;
            for (i = 0; i < P.n_stations; i++) {
                if (!S.has_bottle[i]) n_free++;
                else if (S.V[i] > 0.0) n_full++;
                else n_empty++;
            }
            /* 供給 315 / 充填 0 / 排出 225 なら 満量 5 + 空瓶 1 + 空ホルダ 2 */
            check_int("初期の満量ホルダ数", n_full, residence_indexes(&P) - 1);
            check_int("初期の空瓶（供給を受けたところ）", n_empty, 1);
            check_int("初期の空ホルダ（ボトル無し）", n_free,
                      P.n_stations - residence_indexes(&P));
            check_int("満量 + 空瓶 + 空ホルダ = ステーション数",
                      n_full + n_empty + n_free, P.n_stations);
        }

    }

    /* -------------------------------------------------------------- */
    printf("\n--- 4. トルクと出力（走査。ピークは同時に起きない） ---\n");
    {
        double j = steady_rigid_all(&P);
        double psi_at = 0.0;
        double t_bare = input_torque_peak(&P, j, 0, 0, &psi_at);
        double t_full;
        check("入力軸ピーク（慣性のみ）", t_bare, 0.402900, 1e-3, "Nm");
        check("そのときの psi", psi_at * 180.0 / M_PI, 51.71, 1e-2, "deg");
        t_full = input_torque_peak(&P, j, 1, 1, &psi_at);
        check("入力軸ピーク（摩擦込み）", t_full, 0.715479, 1e-3, "Nm");
        check("引きずりが占める割合", P.input_drag_torque / t_full, 0.41930, 1e-2, "[-]");
        check("ピーク機械出力（慣性のみ）", peak_power(&P, j, 0), 0.717258, 1e-3, "W");
        /* トルク最大と速度最大を掛けた値は過大になる。その事実を数字で残す。 */
        {
            double naive = j * table_alpha_max(&P) * 0.43990 / P.cam_efficiency;
            check("最大×最大（誤った掛け算）", naive, 0.71804, 1e-2, "Nm");
            check("その過大の倍率", naive / t_bare, 1.7822, 1e-2, "[-]");
        }
    }

    /* -------------------------------------------------------------- */
    printf("\n--- 5. スロッシング（満量） ---\n");
    {
        double h = P.target_volume / (M_PI * P.bottle_R * P.bottle_R);
        double w1 = twin_slosh_omega(&P, h);
        double f1 = w1 / (2.0 * M_PI);
        double L1 = P.g / (w1 * w1);
        double ratio = twin_slosh_mass_ratio(&P, h);
        double half = P.zeta * f1;
        double f0 = 1.0 / P.tact;
        double n = floor(f1 / f0 + 0.5);
        double detune = fabs(f1 - n * f0);
        check("満量の液深", h * 1e3, 120.544, 1e-4, "mm");
        check("固有角周波数 w1", w1, 23.5705, 1e-4, "rad/s");
        check("固有振動数 f1", f1, 3.75136, 1e-4, "Hz");
        check("周期 T1", 1.0 / f1, 0.26657, 1e-4, "s");
        check("等価振り子長 L1", L1 * 1e3, 17.652, 1e-4, "mm");
        check("揺れる質量比 m1/m_liq", ratio, 0.12254, 1e-3, "[-]");
        check("揺れる質量 m1", ratio * P.rho * P.target_volume, 0.04901, 1e-3, "kg");
        check("共振の半値半幅", half, 0.018757, 1e-3, "Hz");
        check_int("最寄り高調波の次数", (long)n, 11);
        check("離調", detune, 0.08469, 1e-3, "Hz");
        check("離調 / 半値幅", detune / half, 4.515, 1e-3, "[-]");
        /* 分裂幅は omega の関数で、定数として持っていないこと */
        check("割出しピークでの分裂 om/(2pi)",
              table_omega_max(&P) / (2.0 * M_PI), 0.14663, 1e-3, "Hz");
        check("同 半値幅の何倍か",
              table_omega_max(&P) / (2.0 * M_PI) / half, 7.817, 1e-3, "[-]");
    }

    /* -------------------------------------------------------------- */
    printf("\n--- 6. 割出し 1 回の応答（満量 1 本・静止から・dt=1e-4） ---\n");
    {
        double pk_c, dw_c, pk_n, dw_n;
        index_slosh_response(&P, 0, &pk_c, &dw_c);
        index_slosh_response(&P, 1, &pk_n, &dw_n);
        check("割出し中のピーク傾き", pk_c * 1e3, 68.074, 1e-3, "mrad");
        check("停留に入っての残留",   dw_c * 1e3, 35.30,  2e-3, "mrad");
        check("コリオリ無しの残留",   dw_n * 1e3, 38.453, 2e-3, "mrad");
        check("コリオリを落とすと残留が", dw_n / dw_c, 1.0891, 1e-2, "倍");
        check("ピークはほとんど動かない", fabs(pk_n - pk_c) * 1e3, 0.0015451, 1e-3, "mrad");
        check("dz/R（ピーク）", tan(pk_c), 0.06818, 1e-3, "[-]");
        check_int("適用範囲 0.2 の内側", tan(pk_c) < TWIN_RANGE_DZR ? 1 : 0, 1);
        /* 準静的な傾き a_t/g との比。強制応答が支配的なことの目印 */
        check("準静的な傾き a_t_max/g",
              P.Rp * table_alpha_max(&P) / P.g * 1e3, 44.27, 1e-3, "mrad");
        check("ピーク / 準静的",
              pk_c / (P.Rp * table_alpha_max(&P) / P.g), 1.5377, 1e-2, "[-]");
    }

    /* -------------------------------------------------------------- */
    printf("\n--- 7. 充填と噴流 ---\n");
    {
        double a_noz = M_PI * (P.nozzle_diameter / 2.0) * (P.nozzle_diameter / 2.0);
        double v_noz = P.flow_rate / a_noz;
        double f_jet = P.rho * P.flow_rate * v_noz;
        double t_fill = P.target_volume / P.flow_rate;
        double t_all = P.start_delay + P.valve_open_delay + t_fill + P.valve_close_delay;
        check("ノズル流速", v_noz, 2.1437, 1e-3, "m/s");
        check("噴流の力", f_jet, 0.70743, 1e-4, "N");
        check("噴流 / 満量の液の重さ", f_jet / (P.rho * P.target_volume * P.g),
              0.18034, 1e-3, "[-]");
        check("充填時間", t_fill, 1.21212, 1e-4, "s");
        check("開始から閉じ切りまで", t_all, 1.32212, 1e-4, "s");
        check("停留の余裕", P.dwell - t_all, 0.17788, 1e-3, "s");
        check_int("停留に収まる", (t_all < P.dwell) ? 1 : 0, 1);
        check("満量の頭上空間",
              (P.body_height - P.target_volume / (M_PI * P.bottle_R * P.bottle_R)) * 1e3,
              29.456, 1e-3, "mm");
    }

    /* -------------------------------------------------------------- */
    printf("\n--- 8. 垂直荷重経路 ---\n");
    {
        double mx, my, mabs;
        double h;
        twin_init(&S, &P, TWIN_PRIME_GEOM);
        /* テーブル角 0 の定常。8 方位に均等なら消えるので、残るのは中身の偏りだけ。 */
        twin_step(&S);   /* 1 ステップだけ回して m_bend を作る（t=0、まだ動いていない） */
        mabs = S.m_bend; mx = S.m_bend_x; my = S.m_bend_y;
        check("定常の曲げモーメント", mabs, 2.22529, 1e-4, "Nm");
        check("満量 1 本ぶんの液の重量", P.rho * P.target_volume * P.g * P.Rp,
              0.88260, 1e-4, "Nm");
        /* 8 方位に同じ質量が載っていれば和は消えるので、残るのは「満量からの欠け」だけ。
         * 空ホルダ 2 つはボトルごと 0.425 kg、供給されたばかりの空瓶は液 0.400 kg が欠ける。
         * 位置は 225 / 270 / 315 deg で、45 deg ずつ離れている。 */
        {
            double mm = P.bottle_mass + P.rho * P.target_volume;   /* 満量 1 本 */
            double sx = mm * cos(P.discharge_angle)
                      + mm * cos(P.discharge_angle + P.index_angle)
                      + P.rho * P.target_volume * cos(P.infeed_angle);
            double sy = mm * sin(P.discharge_angle)
                      + mm * sin(P.discharge_angle + P.index_angle)
                      + P.rho * P.target_volume * sin(P.infeed_angle);
            h = hypot(sx, sy);
            check("欠け分の合成質量", h, 1.00851, 1e-4, "kg");
            check("そこから出るモーメント", h * P.g * P.Rp, 2.22529, 1e-4, "Nm");
        }
        check("向きの成分の大きさ", hypot(mx, my), mabs, 1e-12, "Nm");
    }

    /* -------------------------------------------------------------- */
    printf("\n--- 9. ホルダ番号と工程 ---\n");
    {
        double pitch = 2.0 * M_PI / (double)P.n_stations;
        /* 最初の停留（テーブル角 = 1 ピッチ）で、排出 270deg / 供給 315deg に
         * 来るホルダ番号。定常初期状態と辻褄が合っていること。 */
        check_int("排出に来るホルダ（1 停留目）",
                  twin_holder_at(&P, P.discharge_angle, pitch), 4);
        check_int("供給に来るホルダ（1 停留目）",
                  twin_holder_at(&P, P.infeed_angle, pitch), 6);
        check_int("充填に来るホルダ（1 停留目）",
                  twin_holder_at(&P, P.fill_angle, pitch), 7);
        check_int("テーブル角 0 で充填位置", twin_holder_at(&P, P.fill_angle, 0.0), 0);
        /* 1 本のボトルの滞留 = 供給から排出まで（工程角から一意に決まる） */
        check_int("供給から排出までの割出し回数", residence_indexes(&P), 6);
        check("滞留時間", (double)residence_indexes(&P) * P.tact, 18.0, 1e-9, "s");
        /* 供給と排出は 2 ステーション離れている（星車どうしが当たらないため）。
         * 隣り合う 2 角に置くと、両方の星車が抱えるボトルが干渉する。 */
        check_int("供給と排出の間隔（ステーション数）",
                  ((int)nearbyint((P.infeed_angle - P.discharge_angle) / pitch)
                   % P.n_stations + P.n_stations) % P.n_stations, 2);
    }

    /* -------------------------------------------------------------- */
    printf("\n--- 10. 1 タクト回す（定常状態から） ---\n");
    {
        long k, nstep;
        double v_before, v_after;
        int n_full0 = 0, n_full1 = 0;
        twin_init(&S, &P, TWIN_PRIME_GEOM);
        for (i = 0; i < P.n_stations; i++) if (S.V[i] > 0.0) n_full0++;
        nstep = (long)floor(P.tact / P.dt + 0.5);
        v_before = S.V[P.n_stations - 1];
        for (k = 0; k < nstep; k++) twin_step(&S);
        v_after = S.V[P.n_stations - 1];
        check("1 タクトのステップ数", (double)nstep, 120000.0, 1e-9, "[-]");
        check("空瓶に入った量", (v_after - v_before) * 1e6, 400.0, 1e-4, "mL");
        check_int("排出した本数", S.n_discharged, 1);
        check_int("供給した本数", S.n_infeed, 1);
        check("排出した液量", S.out_volume * 1e6, 400.0, 1e-4, "mL");
        check("こぼれ", S.spill_total, 0.0, 1e-15, "m^3");
        for (i = 0; i < P.n_stations; i++) if (S.V[i] > 0.0) n_full1++;
        check_int("満量の本数が 1 タクトで戻る（定常）", n_full1, n_full0);
        check("テーブル角", S.th_t, P.index_angle, 1e-12, "rad");
        check("割出し中の最大傾き", S.max_tilt_index * 1e3, 68.074, 1e-2, "mrad");
        check_int("適用範囲の逸脱なし", S.range_warned, 0);
        /* 通しで拾ったピークは剛体換算より大きい（粘性 + スロッシング反力ぶん）。
         * T_slosh を足す向きにすると逆に小さくなるので、ここが符号の番人になる。
         * 条件は py/ref.py の summary() と同じ（2 タクト・dt = 0.2 ms）。 */
        {
            static twin_state Q;
            twin_params PQ = P;
            double rigid = steady_rigid_all(&P) * table_alpha_max(&P);
            long j;
            PQ.dt = 2.0e-4;
            twin_init(&Q, &PQ, TWIN_PRIME_GEOM);
            for (j = 0; j < 2 * (long)floor(PQ.tact / PQ.dt + 0.5); j++) twin_step(&Q);
            check_int("ピークが剛体換算 J*alpha_max を上回る",
                      (Q.max_t_table > rigid) ? 1 : 0, 1);
            check("通しのピークトルク（2 タクト・dt=0.2ms）", Q.max_t_table, 1.42004,
                  1e-4, "Nm");
            check("剛体換算に対する比", Q.max_t_table / rigid, 1.0235, 1e-3, "[-]");
        }
        /* 停留中は omega = alpha = 0 なので、トルクは厳密に反力だけ（符号は反転して入る）*/
        check("停留末のテーブル軸トルク", S.t_table, -S.t_slosh, 1e-12, "Nm");
        check("停留末の入力軸トルク", S.t_input, P.input_drag_torque, 1e-12, "Nm");
    }

    /* -------------------------------------------------------------- */
    printf("\n--- 11. 軸受のイベント（モータ軸。停留中も打つ）---\n");
    {
        long k, nstep, n_bear = 0, n_dwell = 0;
        twin_params Q = P;
        Q.flt.bear_enabled = 1;
        if (!(Q.flt.bear_defect_freq > 0.0)) Q.flt.bear_defect_freq = 89.5;
        twin_init(&S, &Q, 1);
        nstep = (long)floor(Q.tact / Q.dt + 0.5);
        for (k = 0; k < nstep; k++) {
            int e, ne;
            twin_step(&S);
            ne = twin_event_count(&S);
            for (e = 0; e < ne; e++) {
                double v[5 + TWIN_EV_DATA];
                twin_event_get(&S, e, v);
                if ((int)v[1] == TWIN_EV_BEARING) {
                    n_bear++;
                    if (v[0] - floor(v[0] / Q.tact) * Q.tact >= Q.index_time) n_dwell++;
                }
            }
        }
        check("1 タクトの衝撃回数", (double)n_bear,
              floor(Q.flt.bear_defect_freq * Q.tact) + 1.0, 1e-9, "件");
        check_int("停留中にも出ている", (n_dwell > 100) ? 1 : 0, 1);
        check("モータ 1 回転あたりの衝撃", Q.flt.bear_defect_freq
              / (Q.motor_speed / (2.0 * M_PI)), 3.58, 1e-3, "[-]");
    }

    printf("\n%d 件中 %d 件が不一致\n", n_check, n_fail);
    if (n_fail) {
        printf("NG: 期待値と合わない。**期待値を書き換えて合わせにいかないこと。**\n"
               "    py/ref.py と params.json を見て、どちらが動いたのかを先に特定する\n");
        return 1;
    }
    printf("OK: すべて期待値の内側\n");
    return 0;
}
