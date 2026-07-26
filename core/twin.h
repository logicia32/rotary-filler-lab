/* twin.h -- ロータリー充填機 仮想モデル / 物理コア（params.json rev.3）
 *
 * 機械はカム式インデックスユニット＋誘導ギヤモータ。位置ループも速度ループも
 * 電流ループも無い。テーブル角はカム入力軸角 psi の関数で、時間の関数ではない。
 *
 * **仕様書は py/ref.py。** 式・符号・積分順序はすべてそちらに合わせてある。
 * MODEL.md と食い違うところは FORMAT.md と README.md に名指しで書いた。
 * 数値は params.json から読む。C99、外部ライブラリ依存なし（libm のみ）。malloc は使わない。
 */
#ifndef TWIN_H
#define TWIN_H

#include <stddef.h>

/* ステーション数の上限（固定配列） */
#define TWIN_MAX_STATIONS 32

/* 1 レコードの構成（FORMAT.md 2 節が正。py/ref.py の Record と同じ並び）
 *   スカラ 15     : t, psi, th_t, omega, alpha, th_m, j_load,
 *                   torque_table, torque_input, torque_slosh,
 *                   m_bend, m_bend_x, m_bend_y, f_tab_t, f_tab_r
 *   ステーション毎 5: V, h, phi_t, phi_r, spill
 */
#define TWIN_SCALARS_PER_REC   15
#define TWIN_FIELDS_PER_ST     5
#define TWIN_MAX_REC_FLOATS    (TWIN_SCALARS_PER_REC + TWIN_FIELDS_PER_ST * TWIN_MAX_STATIONS)

/* 適用範囲（py/ref.py の TILT_WARN_RATIO / TILT_INVALID_RATIO）。
 * 壁面の液面上下 dz = R*tan(tilt) が R のこの比を超えると、
 * 波が砕けたり回り始めて 1 次モードの線形近似が成り立たない。 */
#define TWIN_RANGE_WARN_DZR 0.1
#define TWIN_RANGE_DZR      0.2

/* J1'(x) = 0 の第 1 根。厳密値 1.8411837813406595 との差は相対 8.8e-6。
 * py/ref.py の EPS1 と同じ値でなければ突き合わせが成立しない。 */
#define TWIN_EPS1 1.8412

/* ------------------------------------------------------------------ */
/* イベント（連続量に乗せないもの。FORMAT.md 5 節）                    */
/*   軸受のリンギング 3 kHz は連続ログ 4 kHz のナイキストを割るので、    */
/*   衝撃は波形として連続ログに乗せず、発生時刻と振幅の列として別に出す。 */
/* ------------------------------------------------------------------ */
#define TWIN_EV_BEARING        0   /* 軸受外輪傷（モータ軸）。d0 加速度 / d1 リンギング周波数 / d2 減衰比 */
#define TWIN_EV_CAM_IMPACT     1   /* カムフォロワの当たり。d0 トルク / d1 すきま */
#define TWIN_EV_INFEED         2   /* 供給。d0 空瓶の質量 */
#define TWIN_EV_INFEED_MISSED  3   /* 欠品で供給されなかった */
#define TWIN_EV_DISCHARGE      4   /* 排出。d0 体積 / d1 傾き / d2 phi_t / d3 phi_r / d4 dphi_t / d5 dphi_r / d6 質量 */
#define TWIN_EV_VALVE_DRIP     5   /* 弁の液垂れ。d0 体積 */

#define TWIN_EV_DATA    8          /* 種別ごとの値の枠 */
#define TWIN_MAX_EVENTS 16         /* 1 ステップで同時に立ちうる件数の上限 */

/* 1 サイクルぶんの予定イベント（カムフォロワ 2 件 + 軸受）の上限。
 * 軸受は 89.5 Hz x タクト 3.0 s = 269 件。余裕を見て 512。 */
#define TWIN_MAX_SCHED 512

typedef struct {
    double t;                  /* 発生時刻 [s] */
    double th_t;               /* そのときのテーブル角 [rad] */
    double th_m;               /* そのときのモータ軸角 [rad] */
    double d[TWIN_EV_DATA];    /* 種別ごとの値 */
    int    kind;               /* TWIN_EV_* */
    int    station;            /* 対象ホルダ番号。機械全体の事象なら -1 */
} twin_event;

/* ------------------------------------------------------------------ */
/* 故障モード（params.json の faults セクション。既定は全て無効）      */
/*   rev.3 で入れ替わっている。充填量アンバランスとバックラッシュは     */
/*   この機構では成立しないので無い。                                  */
/* ------------------------------------------------------------------ */
typedef struct {
    /* 欠品: 供給スターホイールの取りこぼしで、指定ホルダだけボトルが載らない */
    int    missing_enabled;
    int    missing_station;

    /* 弁の閉じ遅れ: 全ボトルが等しく過充填になり、液垂れが落ちる */
    int    valve_enabled;
    double valve_extra_delay;   /* 追加の閉じ遅れ [s] */
    double valve_drip_volume;   /* 液垂れ体積 [m^3] */

    /* カムフォロワ摩耗: 割出しの入口と出口で当たりが出る */
    int    cam_enabled;
    double cam_clearance;       /* すきま [rad] */
    double cam_impact_torque;   /* 衝撃トルク [Nm] */

    /* 軸受外輪傷（モータ軸） */
    int    bear_enabled;
    double bear_defect_freq;    /* 欠陥通過周波数 [Hz] */
    double bear_accel;          /* 衝撃の振幅 [m/s^2] */
    double bear_ring_freq;      /* リンギング周波数 [Hz] */
    double bear_ring_damping;   /* リンギングの減衰比 [-] */
} twin_faults;

/* ------------------------------------------------------------------ */
/* パラメータ（params.json を SI に正規化して保持）                    */
/* ------------------------------------------------------------------ */
typedef struct {
    /* ボトル */
    double bottle_R;            /* 内半径 [m] */
    double body_height;         /* 胴部高さ = 液が入りうる高さの上限 [m] */
    double bottle_mass;         /* 空瓶質量 [kg] */

    /* 液 */
    double rho;                 /* 密度 [kg/m^3] */
    double nu;                  /* 動粘度 [m^2/s]（参考値） */
    double zeta;                /* スロッシング減衰比 [-] */

    /* テーブル */
    double Rp;                  /* ピッチ円半径 [m] */
    int    n_stations;          /* ステーション数 */
    double index_angle;         /* 1 回の割出し角 [rad] */
    double bare_inertia;        /* 空テーブルの極慣性 [kg m^2] */
    /* 板の寸法は bare_inertia の検算にしか使わない（単体試験）。 */
    double plate_diameter;      /* 板の外径 [m] */
    double plate_thickness;     /* 板厚 [m] */
    double plate_density;       /* 板の密度 [kg/m^3] */
    double plate_mass;          /* 板の質量 [kg]（json の値） */

    /* ステーション（世界角 [rad]） */
    double infeed_angle;
    double fill_angle;
    double discharge_angle;

    /* カム索引ユニット */
    double index_angle_input;   /* 割付角（入力軸）[rad] */
    double dwell_angle_input;   /* 停留角（入力軸）[rad] */
    double cam_efficiency;      /* 効率 [-]。**仮置き**（出所なし） */
    double input_drag_torque;   /* 入力軸の引きずりトルク [Nm]。**仮置き** */
    double table_viscous;       /* テーブル粘性 [Nm s/rad]。**仮置き** */

    /* サイクル */
    double tact;                /* タクト [s] */
    double index_time;          /* 割出し時間 [s] */
    double dwell;               /* 停留時間 [s] = tact - index_time（導出値） */

    /* 駆動系 */
    double gear_ratio;          /* 減速比 [-] */
    double motor_speed;         /* モータ軸角速度 [rad/s]（運転点） */

    /* 充填 */
    double target_volume;       /* 目標吐出量 [m^3] */
    double flow_rate;           /* 流量 [m^3/s] */
    double nozzle_diameter;     /* ノズル内径 [m] */
    double valve_open_delay;    /* 弁の開き遅れ [s] */
    double valve_close_delay;   /* 弁の閉じ遅れ [s] */
    double start_delay;         /* 停留開始から開指令までの待ち [s] */

    /* シミュレーション */
    double dt;                  /* 刻み [s] */
    double log_rate;            /* ログ周波数 [Hz] */
    double frame_rate;          /* 参考（描画側が使う）[Hz] */
    double g;                   /* 重力加速度 [m/s^2] */

    /* 取り決め（params.json に無いのでこちらで決めた値） */
    double jet_fall_height;     /* ノズル出口から液面までの落差 [m]。既定 0 */
    int    no_coriolis;         /* 1 なら回転基底の 3 項を落とす（比較用。既定 0） */

    twin_faults flt;
} twin_params;

/* ------------------------------------------------------------------ */
/* 状態                                                                */
/* ------------------------------------------------------------------ */
typedef struct {
    twin_params p;              /* 値で持つ（ctypes から 1 個の塊として扱えるように） */

    /* --- 進行 --- */
    long   step;                /* 通算ステップ数 */
    long   k;                   /* サイクル内ステップ番号 */
    long   n_steps_cycle;       /* 1 サイクルのステップ数 = round(tact/dt) */
    int    cycle;               /* サイクル番号（0 始まり） */
    double t;                   /* いま計算しているステップの時刻 [s] */
    double t0;                  /* このサイクルの先頭時刻 [s] */
    double th_base;             /* 割出しの積み上げぶんのテーブル角 [rad] */
    double next_log;            /* 次にログを書く時刻 [s] */
    double log_interval;        /* ログ間隔 [s] */
    int    do_log;              /* このステップを記録するか */

    /* --- サイクル内の段取り --- */
    int    i_out;               /* このサイクルで排出するホルダ番号 */
    int    i_in;                /* このサイクルで供給するホルダ番号 */
    int    transferred;         /* 受け渡しを済ませたか */
    int    closing;             /* 弁の閉じ命令を出したあとか */
    double close_t;             /* 閉じ命令を出した時刻 [s] */
    double valve_open_t;        /* 流出が始まる時刻 [s] */

    /* --- 割出しの運動 --- */
    double psi;                 /* カム入力軸角 [rad]、[0, 2pi) */
    double th_t, omega, alpha;  /* テーブル角・角速度・角加速度 */
    double th_m;                /* モータ軸角 [rad]（単調増加） */
    double ratio;               /* カムの瞬時変速比 d th_t / d psi [-] */
    int    in_index;            /* 割出し中か */
    double a_t, a_r;            /* ボトル中心の水平加速度（局所成分）[m/s^2] */

    /* --- ホルダ --- */
    int    has_bottle[TWIN_MAX_STATIONS];
    int    filled[TWIN_MAX_STATIONS];    /* 充填済みか */
    double V[TWIN_MAX_STATIONS];         /* 液量 [m^3] */
    double spill[TWIN_MAX_STATIONS];     /* こぼれ積算 [m^3] */
    double phi_t[TWIN_MAX_STATIONS];     /* 振り子角（接線）[rad] */
    double phi_r[TWIN_MAX_STATIONS];     /* 振り子角（半径）[rad] */
    double dphi_t[TWIN_MAX_STATIONS];    /* 回転座標系での角速度 [rad/s] */
    double dphi_r[TWIN_MAX_STATIONS];
    double h[TWIN_MAX_STATIONS];         /* 液深 [m] */
    double w1[TWIN_MAX_STATIONS];        /* 1 次モード固有角周波数 [rad/s] */
    double L1[TWIN_MAX_STATIONS];        /* 等価振り子長 [m] */
    double m0[TWIN_MAX_STATIONS];        /* 一緒に動く液質量 [kg] */
    double m1[TWIN_MAX_STATIONS];        /* 揺れる液質量 [kg] */
    double dzr[TWIN_MAX_STATIONS];       /* dz/R = tan(tilt) [-] */
    int    range_bad[TWIN_MAX_STATIONS]; /* 適用範囲逸脱（ラッチ） */
    int    i_fill;                       /* いま充填ステーションにいるホルダ番号 */
    int    filling;                      /* このステップで注いでいるか */

    /* --- 出力量 --- */
    double j_load;              /* 負荷慣性 [kg m^2]。液は m0 だけ */
    double t_table;             /* テーブル軸トルク [Nm] */
    double t_input;             /* カム入力軸トルク [Nm] */
    double t_slosh;             /* 揺れる液がテーブル軸に返すトルク [Nm] */
    double m_bend, m_bend_x, m_bend_y;   /* 曲げモーメント [Nm] */
    double f_tab_x, f_tab_y;    /* テーブルが受ける水平合力（世界座標）[N] */
    double dj_dt;               /* 充填中の dJ_load/dt [kg m^2/s]。停留中は 0 */

    /* --- 診断 --- */
    double max_tilt;            /* 合成傾きの最大 [rad] */
    double max_tilt_index;      /* 割出し中の最大 [rad] */
    double max_tilt_dwell;      /* 停留中の最大 [rad] */
    double max_dzr;             /* dz/R の最大 [-] */
    double max_t_table;         /* |T_table| の最大 [Nm] */
    double max_t_input;         /* T_in の最大 [Nm]（符号付き） */
    double min_t_input;         /* T_in の最小 [Nm]。減速中は負になる */
    double max_t_slosh;         /* |T_slosh| の最大 [Nm] */
    double max_m_bend;          /* |M| の最大 [Nm] */
    double max_f_tab;           /* 水平合力の大きさの最大 [N] */
    double max_power;           /* |T_table*omega| の最大 [W] */
    int    range_warned;        /* dz/R が 0.2 を一度でも超えたか */
    long   n_discharged;        /* 排出した本数 */
    long   n_infeed;            /* 供給した本数 */
    long   n_missed;            /* 欠品で供給できなかった回数 */
    double out_volume;          /* 排出した液量の積算 [m^3] */
    double spill_total;         /* こぼれた液量の積算 [m^3] */

    /* --- イベント --- */
    twin_event ev[TWIN_MAX_EVENTS];      /* 直前の twin_step で出たぶん */
    int    n_ev;
    long   n_ev_total;
    long   n_ev_lost;                    /* 溢れて捨てた件数（0 であるべき） */
    twin_event sched[TWIN_MAX_SCHED];    /* このサイクルの予定イベント（時刻順） */
    int    n_sched;
    int    i_sched;                      /* 次に出す予定イベント */
} twin_state;

/* ------------------------------------------------------------------ */
/* API                                                                 */
/* ------------------------------------------------------------------ */

/* params.json を読む。成功で 0、失敗で非 0（err に日本語メッセージ）。 */
int  twin_params_load(twin_params *p, const char *path, char *err, size_t errcap);

/* 初期状態の作り方（FORMAT.md 4 節） */
#define TWIN_PRIME_EMPTY 0   /* 全ホルダ空（ボトルも無い） */
#define TWIN_PRIME_GEOM  1   /* 工程配置から作った定常状態。**既定** */

/* 状態を初期化する。prime は TWIN_PRIME_*。 */
void twin_init(twin_state *s, const twin_params *p, int prime);

/* 外から V[] や phi[] を書き換えたあとに、そこから決まる量
 * （液深・固有角周波数・等価振り子長・等価質量・慣性）を作り直す。
 * twin_step はステップの中で同じことをするので、通常は呼ばなくてよい。 */
void twin_refresh(twin_state *s);

/* dt / log_rate の妥当性を確認する。問題があれば msg に連結して返す（0 = 問題なし）。 */
int  twin_check_dt(const twin_state *s, char *msg, size_t cap);

/* 1 ステップ進める。戻った直後の状態が、そのステップの時刻 s->t に対応する。 */
void twin_step(twin_state *s);

/* このステップを記録すべきか（filler と同じ間引き）。 */
int  twin_should_log(const twin_state *s);

/* 現在状態を 1 レコード分の double に詰める。詰めた要素数を返す。 */
int  twin_record(const twin_state *s, double *buf, int cap);

/* 1 レコードの要素数（浮動小数のぶん。在荷フラグは別）。 */
int  twin_record_floats(const twin_state *s);

/* 在荷フラグを u8 の 0/1 で詰める（ステーション数ぶん）。詰めた個数を返す。 */
int  twin_present(const twin_state *s, unsigned char *buf, int cap);

/* 直前の twin_step で発生したイベントの数。 */
int  twin_event_count(const twin_state *s);

/* k 番目のイベントを out[13] = {t, kind, station, th_t, th_m, d0..d7} に取り出す（成功で 0）。
 * 構造体の並びを外から知らずに済むよう、double の配列で渡す。 */
int  twin_event_get(const twin_state *s, int k, double *out);

/* 世界角 world にいるホルダ番号（テーブル角 th_t のとき）。 */
int  twin_holder_at(const twin_params *p, double world, double th_t);

/* 変形正弦カムの無次元 (変位, 速度, 加速度)。x は 0..1。自己検査と単体試験で使う。 */
void twin_modified_sine(double x, double *s, double *v, double *a);

/* 入力軸角 psi のときの (テーブル相対角, 角速度, 角加速度)。 */
void twin_table_motion(const twin_params *p, double psi,
                       double *th, double *omega, double *alpha);

/* カムの瞬時変速比 d th_t / d psi [-]。 */
double twin_table_ratio(const twin_params *p, double psi);

/* 液深 h [m] の 1 次モード固有角周波数 [rad/s]。h <= 0 なら 0。 */
double twin_slosh_omega(const twin_params *p, double h);

/* 液のうち揺れる分の割合 m1/m_liq [-]。h <= 0 なら 0。 */
double twin_slosh_mass_ratio(const twin_params *p, double h);

/* ctypes 用: 構造体サイズ */
size_t twin_state_size(void);

#endif /* TWIN_H */
