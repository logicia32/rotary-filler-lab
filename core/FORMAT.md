# 出力バイナリの仕様（core/filler が吐くファイル）

MODEL.md 10 節と SENSORS.md の実体。読む側（センサ合成・描画）はこの文書だけ見れば足りる。
**この文書が唯一の取り決め。** 並びを変えるときは magic を上げて、ここを先に直す。

版 `RFILL004` は **params.json rev.3（カム式インデックスユニット＋誘導ギヤモータ）** の版。
列の並びは `py/ref.py` の `Record` と 1 対 1 で、そちらが正。この文書はバイト並びだけを決める。

`filler` を 1 回走らせると、ファイルが 2 つ出る。

| ファイル | magic | 中身 | 節 |
|---|---|---|---|
| `--out` で指定したもの | `RFILL004` | 連続量のログ（4 kHz 間引き） | 1〜2 |
| 同じ名前 + `.events` | `RFEVT002` | イベント列（時刻・種別・振幅） | 5 |

分けてある理由は SENSORS.md 5.5 節。軸受のリンギング 3 kHz を連続ログ 4 kHz に乗せると
ナイキストを割って折り返す。**衝撃は波形にせず、いつ・どれだけ、の列で渡す。**
波形への合成はセンサ層が各センサの刻みで行う。
rev.3 では連続ログに衝撃系の列を**一切置いていない**（前版の `a_bear` / `T_bl` は消滅）。

- バイト順は **リトルエンディアン固定**。`filler` は起動時に実行環境がリトルエンディアン
  でなければエラーで止まる。
- 数値型は `u32` = 符号なし 32bit 整数、`i32` = 符号付き 32bit 整数、
  `f32` = IEEE754 単精度、`f64` = IEEE754 倍精度。
- ファイル = **160 バイトのヘッダ 1 個** + **レコードの並び**（隙間・終端マーカ無し）。
- レコード長はヘッダの `record_bytes`。ヘッダを読んでからその長さで刻む。

## 0. 版

| magic | ヘッダ長 | スカラ数 | ステーション毎 | 備考 |
|---|---:|---:|---:|---|
| `RFILL001` | 64 | 6 | 5 | 旧版（サーボ機）。反力・故障モード以前 |
| `RFILL002` | 128 | 13 | 9 | 旧版（サーボ機）。カスケード制御・バックラッシュ前提 |
| `RFILL003` | 160 | 15 | 5 | 2026-07-23 の途中版。**残っていない**（下の 004 に置き換え） |
| `RFILL004` | 160 | 15 | 5 + 在荷 u8 | **現行**。カム式。003 から水平力の名前・符号・並びが変わり、在荷フラグが増えた |

`RFILL002` とはヘッダ長もレコード長も**列の意味も**違う。magic で分岐すること。
002 の読み方で 003 を読むと、`th_t` の位置に `psi` が来るので静かに間違う。

**`sensors/read_dump.py` は 001 / 002 しか知らない**（rev.3 の書き直しの対象外だった）。
003 を読むには版を足すこと。`core/` 側のチェックスクリプトは `core/dumpio.py` を使う。

## 1. ヘッダ（160 バイト）

| offset | size | 型 | 名前 | 内容 |
|---:|---:|---|---|---|
| 0  | 8 | char[8] | `magic` | `"RFILL004"`（終端 NUL 無し） |
| 8  | 4 | u32 | `header_bytes` | 160。データはここから始まる |
| 12 | 4 | u32 | `record_bytes` | `elem_bytes * (n_scalars + n_per_station * n_stations) + flag_bytes` |
| 16 | 4 | u32 | `n_stations` | ステーション数（`table.stations`） |
| 20 | 4 | u32 | `n_scalars` | レコード先頭のスカラ数 = 15 |
| 24 | 4 | u32 | `n_per_station` | ステーション 1 個あたりの要素数 = 5 |
| 28 | 4 | u32 | `elem_bytes` | 1 要素のバイト数。**4 = f32（既定） / 8 = f64（`--f64`）** |
| 32 | 4 | u32 | `n_records` | レコード数。**書き込み完了時に書き戻す** |
| 36 | 4 | u32 | `fault_flags` | bit0 欠品 / bit1 弁閉じ遅れ / bit2 カムフォロワ摩耗 / bit3 軸受外輪傷 |
| 40 | 8 | f64 | `dt_s` | 積分刻み [s] |
| 48 | 8 | f64 | `log_dt_s` | ログ間隔 [s] = `1 / log_rate_hz`（公称値） |
| 56 | 4 | f32 | `pitch_radius_m` | ピッチ円半径 `Rp` [m] |
| 60 | 4 | f32 | `bottle_radius_m` | ボトル内半径 `R` [m] |
| 64 | 4 | f32 | `body_height_m` | 胴部高さ = こぼれ判定の上限 [m] |
| 68 | 4 | f32 | `max_tilt_rad` | 全ステーション・全時間の合成傾きの最大 [rad]。**書き戻し** |
| 72 | 4 | f32 | `max_dz_over_R` | 同 `dz/R = tan(tilt)` の最大 [-]。**書き戻し** |
| 76 | 4 | u32 | `range_exceeded` | 一度でも `dz/R > range_limit` になったか（0/1）。**書き戻し** |
| 80 | 4 | f32 | `range_limit` | 適用範囲のしきい値。0.20（傾き 11.31 deg 相当） |
| 84 | 4 | f32 | `tact_s` | タクト [s] |
| 88 | 4 | f32 | `index_time_s` | 割出し時間 [s] |
| 92 | 4 | f32 | `dwell_s` | 停留時間 [s] = `tact - index_time`（導出値） |
| 96 | 4 | f32 | `index_angle_rad` | 1 回の割出し角 [rad] = `2*pi/N` |
| 100| 4 | f32 | `input_shaft_speed_rad_s` | カム入力軸の角速度 [rad/s] = `2*pi/tact` |
| 104| 4 | f32 | `motor_speed_rad_s` | モータ軸の角速度 [rad/s]（運転点） |
| 108| 4 | f32 | `gear_ratio` | 減速比 [-] |
| 112| 4 | f32 | `target_volume_m3` | 目標吐出量 [m^3] |
| 116| 4 | f32 | `flow_rate_m3_s` | 流量 [m^3/s] |
| 120| 4 | f32 | `w1_full_rad_s` | 満量時の 1 次モード固有角周波数 [rad/s]。既定値では 23.5705（3.7514 Hz） |
| 124| 4 | f32 | `infeed_angle_rad` | 供給の世界角 [rad]（既定 315 deg） |
| 128| 4 | f32 | `fill_angle_rad` | 充填の世界角 [rad]（既定 0 deg） |
| 132| 4 | f32 | `discharge_angle_rad` | 排出の世界角 [rad]（既定 225 deg） |
| 136| 4 | f32 | `cam_efficiency` | カム効率 [-]。**仮置き**（0.85） |
| 140| 4 | f32 | `input_drag_torque_Nm` | 入力軸の引きずりトルク [N m]。**仮置き**（0.30） |
| 144| 4 | f32 | `table_viscous_Nms_rad` | テーブル粘性 [N m s/rad]。**仮置き**（0.05） |
| 148| 4 | f32 | `bearing_defect_freq_hz` | 軸受外輪傷の通過周波数 [Hz]。故障無効なら 0 |
| 152| 4 | i32 | `missing_station` | 欠品故障の対象ホルダ。無効なら -1 |
| 156| 4 | u32 | `flag_bytes` | レコード末尾の在荷フラグの領域 [byte]。`n_stations` 個の u8 を `elem_bytes` の倍数へ切り上げたもの（既定 8） |

`n_records` / `max_tilt_rad` / `max_dz_over_R` / `range_exceeded` は計算が終わってから
`fseek` で書き戻す。**出力先が `-`（標準出力）のときは書き戻せないので、この 4 つは 0 のまま**になる。
`n_records` が 0 のときはファイルサイズから `(size - header_bytes) / record_bytes` で数える。

摩擦の 3 つ（`cam_efficiency` / `input_drag_torque_Nm` / `table_viscous_Nms_rad`）を
ヘッダに入れてあるのは、**仮置きの値が結果のどこにどれだけ効いたかを、後から
データだけで追えるようにするため**。入力軸トルクはこの 3 つで 3 割から 4 割動く。

## 2. レコード

各レコードは **浮動小数の並び + 末尾の在荷フラグ（u8）**。

```
[ elem_bytes x (n_scalars + n_per_station*n_stations) ][ flag_bytes ]
```

既定（`n_stations = 8`、f32）なら 15 + 5*8 = 55 個 = 220 バイト ＋ 在荷 8 バイト
= **228 バイト**。`flag_bytes` は `elem_bytes` の倍数へ切り上げてあるので、
浮動小数の並びは常に揃った境界から始まる。

`log_rate_hz = 4000` なら 912 kB/s、タクト 3.0 s あたり 2.74 MB。
長時間流すときは `--log-rate` を落とすか、読む側でメモリマップ（`np.memmap`）を使うこと。

`--f64` を付けると同じ並びを f64 で書く（レコード長は倍の 440 バイト）。
**`py/ref.py` との突き合わせ用**で、f32 の丸め（相対 6e-8）を挟まずに比べるためにある。
通常のデータ生成には要らない。

### 2.1 先頭のスカラ 15 個

`py/ref.py` の `Record` の並びそのもの。順番を変えないこと。

| index | 名前 | 単位 | 内容 |
|---:|---|---|---|
| 0 | `t` | s | 時刻。サイクル頭からの積み上げ（`cycle*tact + k*dt`） |
| 1 | `psi` | rad | **カム入力軸角。`[0, 2*pi)` に畳んである**。`0 <= psi < index_angle_input` が割出し |
| 2 | `th_t` | rad | テーブル角。0 から単調増加（1 サイクルで `index_angle` 進む） |
| 3 | `omega` | rad/s | テーブル角速度。停留中は厳密に 0 |
| 4 | `alpha` | rad/s^2 | テーブル角加速度。停留中は厳密に 0 |
| 5 | `th_m` | rad | **モータ軸角**。`motor_speed * t` で単調増加。軸受故障の置き場 |
| 6 | `j_load` | kg m^2 | テーブル側の負荷慣性。**液は `m0` だけ**（2.4 節） |
| 7 | `torque_table` | N m | テーブル軸トルク（2.5 節） |
| 8 | `torque_input` | N m | カム入力軸トルク（2.6 節）。**符号付き**（減速中は負になりうる） |
| 9 | `torque_slosh` | N m | 揺れる液が**テーブルに及ぼす**トルク（2.4 節）。`torque_table` には既に引かれて入っている |
| 10 | `m_bend` | N m | テーブル軸の曲げモーメントの大きさ = `hypot(m_bend_x, m_bend_y)` |
| 11 | `m_bend_x` | N m | 同 世界座標 x 成分（2.7 節） |
| 12 | `m_bend_y` | N m | 同 世界座標 y 成分 |
| 13 | `f_tab_x` | N | テーブルが受ける水平合力の**世界座標 x 成分**（2.3 節） |
| 14 | `f_tab_y` | N | 同 **y 成分** |

**消えた列**（002 にあったもの）: `th_cmd`（指令角）、`i_q` / `motor_current`（電流）、
`motor_torque`、`T_bl` / `bl_slip`（バックラッシュ）、`a_bear`（軸受の折り返し波形）。
機構が変わって存在しなくなったか、イベント列へ移った。

**足した列**: `psi`、`th_m`、`torque_input`、`m_bend` / `m_bend_x` / `m_bend_y`、
ステーション毎の `present`（2.2.2 節）。

### 2.2 続いてステーション `i = 0 .. n_stations-1` の順に 5 個ずつ

| offset（ステーション内） | 名前 | 単位 | 内容 |
|---:|---|---|---|
| +0 | `V_i` | m^3 | ボトル内の液量。**mL にするには 1e6 倍** |
| +1 | `h_i` | m | 液深 = `V_i / (pi*R^2)` |
| +2 | `phi_t_i` | rad | 振り子角（接線方向）。**正 = 回転の進む向きに液面が高い** |
| +3 | `phi_r_i` | rad | 振り子角（半径方向）。正 = 外側が高い |
| +4 | `spill_i` | m^3 | こぼれた量の積算（単調増加。受け渡しでもリセットしない） |

要素 `j` のステーション `i` の絶対 index は `n_scalars + n_per_station*i + j`。

合成傾きは `tilt = sqrt(phi_t^2 + phi_r^2)`、壁での上下動は `dz = R*tan(tilt)`。
`dz/R` は読む側で `tan(tilt)` として作る（列としては持たない）。
適用範囲を外れたかどうかはヘッダの `max_dz_over_R` / `range_exceeded` で見る。

### 2.2.1 `phi` の符号（絵にすると必ず引っかかる）

運動方程式 `phi'' + 2*zeta*w1*phi' + w1^2*phi = -a/L1` を定常で解くと、`L1 = g/w1^2` なので

```
phi = -a / g
```

になる。つまり**加速度と逆符号**。ボトルが進行方向へ加速されると液は後ろに寄るが、
そのとき `phi_t` は**負**になる。したがって

- `phi_t > 0` = 液面が**回転の進む向き側**で高い（減速中に起きる）
- `phi_r > 0` = 液面が**外側**で高い（回っている間ずっとこちら。遠心力ぶん）

描画で液面を傾けるときは、この符号のまま `tilt_t = phi_t` として使えばよい。

**回転座標系で積分している。** 接線方向と半径方向は独立ではなく、コリオリ
（`2*omega*phi'`）・オイラー（`alpha*phi`）・遠心（`omega^2*phi`）で結合する。
その結果として 1 次モードが `w1 ± omega` の 2 本に割れるが、**分裂幅は入力ではなく結果**で、
どこにも定数として持っていない。

### 2.2.2 在荷フラグ（レコード末尾の u8）

浮動小数の並びのあとに、**ステーション 1 個につき 1 バイト**の `present` が続く。

| 値 | 意味 |
|---:|---|
| 1 | ホルダにボトルが載っている（**液が 0 の空瓶も 1**） |
| 0 | ホルダにボトルが無い（排出済みで、まだ供給に来ていない） |

`V_i = 0` だけでは「空瓶が載っている」と「ボトルが無い」を区別できないので、
列として持つ。既定条件の定常では 1 が 6 個、0 が 2 個。

`flag_bytes` は `n_stations` を `elem_bytes` の倍数へ切り上げた値で、
余った詰め物は 0。読む側は先頭 `n_stations` バイトだけ見ればよい。

### 2.3 `f_tab_x` / `f_tab_y`（世界座標・テーブルが受ける側）

ステーションごとの接線・半径の向きはテーブル上の位置で違うので、
**局所量をそのまま足しても水平合力にはならない**（8 本が同じ状態なら向きが打ち消し合って 0）。

コアは各ステーションの**反作用**（テーブルが受ける側、2.4 節の `F_react`）

```
F_react = -(m0*a_tank + m1*a1)                       液がホルダに返す力
loc_t   = F_react_t - m_empty*a_t                    空瓶の慣性反力も同じ向きに揃える
loc_r   = F_react_r - m_empty*a_r
F_x    += loc_r*cos(ang_i) - loc_t*sin(ang_i)
F_y    += loc_r*sin(ang_i) + loc_t*cos(ang_i)        ang_i = th_t + i*2*pi/N
```

を世界座標へ回してからベクトル和を取る。出力は **世界座標の x / y 成分そのもの**。
回転基底（接線・半径）へは分解しない。センサの取り付け方位が未決なので、
世界座標のまま出してセンサ層に任せる。

> 版 003 では `f_tab_t` / `f_tab_r` という名前で中身が世界の y / x、しかも符号が
> 逆（液を動かすのに要る力の側）だった。004 で名前・順番・符号をまとめて直した。

- 全ホルダが同じ状態なら `f_tab_* ≈ 0`。**残るのはステーション間のばらつきだけ**。
- 欠品（`faults.missing_bottle`）は、この合力に**回転同期（1 次）成分**として現れる。
- テーブル板そのものは重心が軸上なので水平合力に寄与しない。
- **液が入っていないホルダは、空瓶の慣性反力も数えていない**（`py/ref.py` が
  `volume <= 0` のホルダを丸ごと飛ばしているのに合わせた）。空瓶 1 本ぶん
  （0.025 kg x 0.434 m/s^2 = 0.011 N）が抜ける。既知の食い違い。
- **この力を支持部の曲げに直すには腕の長さが要る**（`L_arm`）。取り付けが未決なので
  `m_bend`（2.7 節）には入れていない。既定条件では
  `|F_tab|` 最大 0.499 N、仮に `L_arm = 0.150 m` を掛けると 0.075 N m で、
  重量経路 2.38 N m の **1/31.8**。

### 2.4 `torque_slosh` と `j_load` の関係（二重計上に注意）

液は `m1`（揺れる分）と `m0`（一緒に動く分）に分けてある。

```
m1 = m_liq * 2R*tanh(eps1*h/R) / (eps1*(eps1^2-1)*h)      eps1 = 1.8412
m0 = m_liq - m1
a1 = -(g*phi + 2*zeta*w1*L1*phi')      ... 運動方程式を代入すると a_tank が消える
torque_slosh = -Rp * sum_i( m1_i * a1_t_i )
```

`torque_slosh` の符号は **「テーブルが受ける側」**（MODEL.md 2.4 の `T_slosh`）。
`F_react`（2.3 節）と同じ向きに揃えてある。

- **`j_load`（index 6）は `m0` だけを数えた慣性。** ボトルが載っていないホルダは 0、
  空瓶だけなら 0.025 kg、液があれば `0.025 + m0`。既定条件の定常で 0.70664 kg m^2。
- 諸元表に出てくる **0.71904 kg m^2 は「液を全量剛体として数えた」別の量**で、
  こちらはログに出てこない。剛体換算のピークトルク 1.38748 N m はその 0.71904 から出る値。
- `torque_slosh` を `j_load` に足し戻すことはしないこと。二重計上になる。
- **`torque_table` には `torque_slosh` が既に引かれて入っている**（2.5 節）。
  ログから足し直さないこと。

既定条件の目安: 満量 400 mL で `m_liq = 0.400 kg`、`m1 = 0.049 kg`（12.25 %）、`m0 = 0.351 kg`。

### 2.5 `torque_table`

```
torque_table = j_load*alpha + (dJ_load/dt)*omega + table_viscous*omega - torque_slosh
```

**`torque_slosh` は引く。** テーブルが受ける側の符号で定義してあるので、
駆動側に要るトルクにするには符号を返す。準静的極限（液が容器に追従して
`a1_t = Rp*alpha`）では `-torque_slosh = +m1*Rp^2*alpha` となり、
**揺動質量が慣性として素直に足される**。足す向きにすると逆に引かれてしまう。
版 003 は足す向きで、テーブル軸ピークトルクが 5.8 % 小さく出ていた。

`dJ_load/dt` は充填中に液が増えるぶん（`Rp^2 * (rho - dm1/dV) * flow_rate`）。
**公称条件では厳密に 0**（充填は停留中に終わり、停留では `omega = 0`）。
タクトを詰めて充填が割出しにはみ出すと効き始める。

旋回軸受の摩擦トルクは `params.json` に無いので 0。そのぶん、
定速区間が厳密に `c_visc*omega` になるのはモデルの都合であって物理ではない。

| 量 | 値 | 条件 |
|---|---:|---|
| 剛体換算 `J_load(0.71904)*alpha_max` | 1.38748 N m | 液を全量剛体、粘性・反力なし |
| 通しで拾ったピーク | 1.42005 N m | 2 タクト・`dt = 0.2 ms`。剛体換算の 1.0235 倍 |

### 2.6 `torque_input`

```
torque_input = T_cam + input_drag_torque
T_cam = torque_table * r(psi) / cam_efficiency     （torque_table*omega >= 0）
T_cam = torque_table * r(psi) * cam_efficiency     （torque_table*omega <  0）
r(psi) = d th_t / d psi = index_angle * Vc(x) / index_angle_input   （0 以上。最大 0.43990）
```

**効率の掛け方は流れの向きで逆になる**（MODEL.md 4.7）。カムが負荷を駆動している間は
損失ぶん余計に要り、負荷がカムを回している間は損失ぶん減って伝わる。
`r` は常に 0 以上なので、向きを決めるのは `torque_table*omega` の符号。
版 003 は絶対値を取って常に `/eta` だったので、減速側を `1/0.85^2 = 1.38` 倍過大に見ていた。

**符号付き**。減速中は `T_cam` が負になるので、`torque_input` は引きずり 0.30 N m を
下回りうる（既定条件では 0.015 〜 0.723 N m）。符号で回生を読むなら
`T_cam = torque_input - drag` を作ってから見ること。

引きずり 0.30 N m は入力軸ピーク 0.7155 N m の **42 %**。効率 0.85 も同じ桁で効く。
**この 2 つは出所の無い仮置き**なので、入力軸トルクを主張の根拠にしないこと。

モータ軸トルクは `torque_input / gear_ratio` で作れるので列には持っていない。
インバータ出力電流は、無負荷電流とトルク電流係数が `params.json` に無いので**出さない**。

### 2.7 `m_bend_*`（曲げモーメント）

ボトルの重量はピッチ円半径 `Rp` に載る。8 方位に均等なら打ち消し合うが、
空きホルダや充填途中があると偏りが残り、テーブル軸に曲げモーメントとして入る。

```
m_bend_x = Rp * sum_i( F_i * cos(ang_i) )
m_bend_y = Rp * sum_i( F_i * sin(ang_i) )      F_i = m_i*g（鉛直下向き）
m_bend   = hypot(m_bend_x, m_bend_y)
```

`m_i` は**液を全量数えた質量**（`m0`/`m1` の分割は水平の慣性力の話で、重量には関係しない）。
充填中は噴流の鉛直運動量 `F_jet = rho*Q*v` を、充填ステーションの世界角に足す。

**`m_bend_x` / `m_bend_y` は「どちらへ偏っているか」を世界座標で持つための成分**で、
曲げの中立軸はこのベクトルに直交する（`py/ref.py` の `bending_moment_from_loads` の取り決め）。
物理としてのモーメントベクトルは、これを 90 deg 回したもの。ひずみへの換算
（断面係数・ヤング率・ゲージ方位）はセンサ層の仕事なので、ここは軸に働く量まで。

**どこで受けるかは決めていない。** 取り付け位置と荷重経路が未決なので、
水平反力による曲げ（`L_arm * (z x F_tab)`、MODEL.md 6.1 節）は入っていない。
既定条件では重量経路 2.3795 N m に対し、水平合力 0.4989 N に仮の腕 0.150 m を掛けて
0.0748 N m で、比は **31.8 倍**。腕の長さが仮置きなのでこの比も仮置き。

> **`m_bend` と `torque_slosh` を比べないこと。** どちらも N m だが、
> `m_bend` は水平軸まわりの曲げ、`torque_slosh` は z 軸まわりのねじり。
> 比を取ると 51.6 倍になるが、これは別の量どうしの比で、
> 「垂直荷重経路が水平経路の何倍か」を言うなら上の 31.8 倍のほう。

### 2.8 受け渡しで値が飛ぶ

供給（315 deg）と排出（225 deg）は**停留の先頭の 1 ステップで一瞬に起きる**。
星車の当たりは連続量に乗せず、イベント列に時刻だけ出す。

| 量 | 排出で | 供給で |
|---|---|---|
| `V_i` / `h_i` | 0 に落ちる | 0 のまま（空瓶） |
| `phi_t_i` / `phi_r_i`（と角速度） | 0 に落ちる | 0 のまま |
| `present_i` | 1 -> 0 | 0 -> 1 |
| `j_load` | 段差（液 `m0` + 瓶ぶん） | 段差（瓶ぶん） |
| `m_bend_*` | 段差 | 段差 |
| `spill_i` | **そのまま**（積算を続ける） | そのまま |

**排出されたボトルの液は、揺れたまま外界へ持ち出される。**
そのときの体積・傾き・振り子角・角速度はイベントに残る（5 節）。
連続量にはその情報が残らないので、必要ならイベント列を見ること。

つまり **タクトに同期した段差が必ず入る**。異常ではなく機械の動作なので、
正常データにも同じものが入っている。

## 3. ctypes から直接叩く場合（libtwin.so）

バイナリを介さずに 1 ステップずつ回したいとき用。公開している関数は次の 10 個。

```
int    twin_params_load(twin_params *p, const char *path, char *err, size_t errcap);
void   twin_init(twin_state *s, const twin_params *p, int prime);
void   twin_refresh(twin_state *s);
int    twin_check_dt(const twin_state *s, char *msg, size_t cap);
void   twin_step(twin_state *s);
int    twin_should_log(const twin_state *s);
int    twin_record(const twin_state *s, double *buf, int cap);
int    twin_record_floats(const twin_state *s);
int    twin_present(const twin_state *s, unsigned char *buf, int cap);
int    twin_event_count(const twin_state *s);
int    twin_event_get(const twin_state *s, int k, double *out);
size_t twin_state_size(void);
```

- `twin_init(s, p, prime)` の `prime` は `TWIN_PRIME_GEOM`（工程配置から作る定常状態、
  4 節）か `TWIN_PRIME_EMPTY`（空のテーブル）。
- `twin_step` は 1 ステップぶんの物理を全部片付けてから、内部のステップ番号を進める。
  **戻った直後の状態が、そのステップの時刻 `s->t` に対応する。**
  `twin_should_log(s)` が真のステップだけ記録すれば、`filler` と同じ間引きになる。
- `twin_record` は 2 節の並びで `double` に詰める（`twin_record_floats` 個）。
  在荷フラグは `twin_present` で別に取る（u8 の 0/1、`n_stations` 個）。
  f32 への丸めは書き出し側の仕事。
- イベントは **直前の `twin_step` で発生したぶんだけ** 溜まっている。
  `twin_event_count()` が件数、`twin_event_get(s, k, out)` が `out[13]` に
  `{時刻, 種別, ホルダ番号, テーブル角, モータ角, d0..d7}` を詰める（5.2 節）。
  次の `twin_step` で消えるので、毎ステップ引き取ること。

`twin_params` / `twin_state` の中身を Python 側で再定義する必要は無い。
`twin_state_size()` で貰ったバイト数の `ctypes.create_string_buffer` を確保して
そのポインタを渡し、値の取り出しは `twin_record()` に任せる。

故障モードを ctypes から切り替えたい場合は、`twin_params` の中身を直接触るのではなく
params.json 側の `faults.*.enabled` を書き換えてから `twin_params_load` を呼ぶこと。

## 4. 初期状態（`prime`）

既定は**工程配置から作る**。ホルダ番号 `i` の世界角は `i*pitch`（テーブル角 0 のとき）。
供給からの割出し回数 `k = ((i*pitch - infeed) / pitch) mod N`、
滞留 `n_res = ((discharge - infeed) / pitch) mod N` に対し

| `k` | 状態 |
|---|---|
| 0 | 空のボトル（供給を受けたところ） |
| `0 < k < n_res` | 満量（充填ステーションを過ぎている） |
| `k >= n_res` | ボトル無し（排出済みで、まだ供給に来ていない） |

いまの params.json（供給 315 / 充填 0 / 排出 225 deg）なら `n_res = 6` で

| ホルダ | 状態 | 世界角 |
|---:|---|---:|
| 0..4 | 満量（400 mL） | 0 .. 180 deg |
| 5, 6 | **ボトル無し** | 225, 270 deg |
| 7 | 空のボトル | 315 deg |

**満量 5 本 + 空瓶 1 本 + 空ホルダ 2 つ**。載っている質量は `5*0.425 + 0.025 = 2.150 kg`、
`J_load`（液を全量剛体）= 0.71904 kg m^2、テーブル軸ピークトルク 1.3875 N m。
滞留は 6 割出し = 18 s。

> **供給と排出は 2 ステーション離してある。** 隣り合う 2 角（45 deg）に置くと、
> 供給側と排出側のスターホイールが抱えるボトルどうしが当たる（2026-07-23 の変更）。

`py/ref.py` の `steady_holders()` も同じ式（`k` と `n_res` から作る）になっているので、
両者は一致する。2026-07-23 の一時期だけ必要だった互換モード `--prime-ref` は撤去した。

### 4.1 `--no-prime`

全ホルダ空（ボトルも無い）から始まる。供給から順に埋まっていくので、
定常に入るまで `N` サイクル掛かる。

### 4.2 ホルダ番号

`holder_at(world, th_t) = round((world - th_t) / (2*pi/N)) mod N`。
丸めは**偶数丸め**（Python の `round` に合わせて `nearbyint` を使っている）。

## 5. イベント列（サイドカー `RFEVT002`）

`--out` のファイル名に `.events` を足したものが既定の出力先（`--events` で変更、
`--no-events` で抑止）。

- バイト順はリトルエンディアン固定。
- ファイル = **64 バイトのヘッダ 1 個** + **56 バイトのレコードの並び**。
- レコードは**時刻の昇順**。同じ時刻に複数件出ることがある。

### 5.1 ヘッダ（64 バイト）

| offset | size | 型 | 名前 | 内容 |
|---:|---:|---|---|---|
| 0  | 8 | char[8] | `magic` | `"RFEVT002"` |
| 8  | 4 | u32 | `header_bytes` | 64 |
| 12 | 4 | u32 | `record_bytes` | 56 |
| 16 | 4 | u32 | `n_events` | 件数。**書き込み完了時に書き戻す** |
| 20 | 4 | u32 | `fault_flags` | 連続ログと同じビット割り |
| 24 | 4 | f32 | `bearing_ring_freq_hz` | 軸受衝撃のリンギング周波数 [Hz]。無効なら 0 |
| 28 | 4 | f32 | `bearing_ring_damping` | リンギングの減衰比 [-]。無効なら 0 |
| 32 | 4 | f32 | `bearing_accel_m_s2` | 衝撃の振幅 [m/s^2]。無効なら 0 |
| 36 | 4 | f32 | `bearing_defect_freq_hz` | 欠陥通過周波数 [Hz]。無効なら 0 |
| 40 | 4 | f32 | `cam_impact_torque_Nm` | カムフォロワの当たりのトルク [N m]。無効なら 0 |
| 44 | 4 | f32 | `cam_clearance_rad` | 同 すきま [rad]。無効なら 0 |
| 48 | 8 | f64 | `dt_s` | 積分刻み [s] |
| 56 | 4 | f32 | `duration_s` | 計算時間 [s] |
| 60 | 4 | u32 | — | 予備。0 |

**ここに入るのは「その回に実際に使った値」**。CLI で上書きしたら上書き後の値が入る。
センサ層が波形を合成するのに要る定数はここに揃っているので、`params.json` を読み直さなくてよい。

### 5.2 レコード（56 バイト）

| offset | size | 型 | 名前 | 内容 |
|---:|---:|---|---|---|
| +0  | 8 | f64 | `t` | 発生時刻 [s] |
| +8  | 4 | u32 | `kind` | 5.3 節 |
| +12 | 4 | i32 | `station` | 対象ホルダ番号。機械全体の事象なら -1 |
| +16 | 4 | f32 | `th_t` | そのときのテーブル角 [rad] |
| +20 | 4 | f32 | `th_m` | そのときのモータ軸角 [rad] |
| +24 | 4*8 | f32[8] | `d0..d7` | 種別ごとの値（5.3 節）。使わない枠は 0 |

### 5.3 種別と `d0..d7`

| `kind` | 名前 | 時刻 | `d0..` |
|---:|---|---|---|
| 0 | `bearing_impulse` | `k / defect_freq_hz`（`k = ceil(t0*f)` から。**時間軸で等間隔**） | d0 加速度 [m/s^2] / d1 リンギング周波数 [Hz] / d2 リンギング減衰比 |
| 1 | `cam_impact` | 割出しの入口 `t0` と出口 `t0 + index_time` | d0 衝撃トルク [N m] / d1 すきま [rad] |
| 2 | `infeed` | 停留の先頭 | d0 空瓶の質量 [kg] |
| 3 | `infeed_missed` | 停留の先頭 | 無し（欠品でボトルが載らなかった） |
| 4 | `discharge` | 停留の先頭 | d0 体積 [m^3] / d1 合成傾き [rad] / d2 `phi_t` / d3 `phi_r` / d4 `dphi_t` / d5 `dphi_r` / d6 質量 [kg] |
| 5 | `valve_drip` | 弁が閉じ切った時刻 | d0 液垂れ体積 [m^3] |

**軸受はモータ軸に置いてある。** 入力軸が連続回転するので、テーブルが止まっている
停留中も等間隔で打ち続ける。既定（89.5 Hz、タクト 3.0 s）で **1 タクトあたり 268 件**。
前版はテーブル軸に置いていて 1 サイクル 0.44 回だったので、そこが決定的に違う。
`th_m` にそのときのモータ軸角が入っているので、位相を使う解析はそれを見ること。

**受け渡し（`infeed` / `discharge`）の当たりの大きさは `params.json` に無い。**
振幅は入れていない。時刻と、持ち出された液の状態だけを渡す。

### 5.4 センサ層はこれをどう使うか

軸受とカムフォロワについて、自分の刻みで

```
a(t) = d0 * exp(-2*pi*ring_damping*ring_freq*(t - t_ev)) * sin(2*pi*ring_freq*(t - t_ev))
```

を作って足す（`t >= t_ev` のところだけ）。合成はリンギングの十分上の刻みで行い、
そこから各センサの帯域制限と間引きを通す。こうすると低速センサでは正しく落ち、
高速センサでは正しく残る。

## 6. CLI

```
filler [--params <path>] [--out <path>|-] [--events <path>] [--no-events]
       [--duration <s>] [--cycles <n>] [--log-rate <hz>] [--dt <s>]
       [--tact <s>] [--index-time <s>] [--jet-fall <m>] [--f64] [--quiet]
       [--no-prime]
       [--fault-missing] [--missing-station <i>]
       [--fault-valve] [--valve-extra-delay <s>] [--drip-mL <x>]
       [--fault-cam] [--cam-torque <Nm>] [--cam-clearance-deg <d>]
       [--fault-bearing] [--bearing-accel <a>] [--bearing-freq <hz>]
       [--no-faults]
```

故障モードは既定で全て無効（`params.json` の `faults.*.enabled` が初期値、そちらも既定は false）。
`--fault-*` を付けると有効になり、`--no-faults` は params.json 側で有効でも全部落とす。
数値オプションを指定すると、その故障は自動的に有効になる。

サマリは標準エラーへ出る。`--out -` で標準出力へ流せばそのままパイプできる。

`make test` で単体試験と `py/ref.py` との突き合わせが走る。手順は `core/README.md`。

## 7. 読む側が引っかかりやすい所

- `V_i` は m^3。mL と 6 桁ずれる。
- `psi` は畳んである（`[0, 2*pi)`）。単調増加ではない。サイクルの境目で 2*pi から 0 へ落ちる。
  **`th_m` のほうは単調増加**。
- `f_tab_x` / `f_tab_y` は**世界座標**。回転基底（接線・半径）ではない（2.3 節）。
  符号は**テーブルが受ける側**で、版 003 とは逆。
- `torque_input` は符号付き。減速中は引きずりを下回る（2.6 節）。
- `torque_table` には `torque_slosh` が**引いて**入っている。足し直さないこと（2.5 節）。
- `j_load` は液の `m0` だけ。諸元表の 0.71904 とは別の量（2.4 節）。
- 在荷はレコード末尾の u8（2.2.2 節）。`V_i = 0` だけでは空瓶と欠品を区別できない。
- `spill_i` は積算。ある区間のこぼれ量は差分を取る。受け渡しでもリセットしない。
- **受け渡しのたびに値が飛ぶ**（2.8 節）。微分を取る処理にはタクト周期の段差として必ず入る。
- 衝撃は連続量に一切乗っていない。**イベント列（5 節）が唯一の出どころ。**
- ログは間引いてある。`log_dt_s` は公称値。コアは `t >= next_log` で書き、
  `next_log` を `1/log_rate` ずつ足していく。`dt` が `log_dt` を割り切らないと刻みが揺れる。
  既定（`dt = 2.5e-5`、`log_rate = 4000`）はちょうど 10 ステップで割り切れる。
- **適用範囲は割出し時間で決まる。** いまの params.json（割出し 1.5 s）なら
  `dz/R` は 0.094 程度で限界 0.20 の内側。`--index-time` を詰めるとすぐ外れる。
  どこで外れるかを見るのが解析の主題なので、外れること自体は目的のうち。
  ただし外れた区間のスロッシングの値はそのまま信じないこと。

### numpy での読み方（例）

```python
import numpy as np

raw = np.memmap(path, dtype='u1', mode='r')
assert bytes(raw[:8]) == b'RFILL004'
hb, rb, nst, nsc, nps, eb, nrec, flags = np.frombuffer(raw[8:40].tobytes(), '<u4')
dt, log_dt = np.frombuffer(raw[40:56].tobytes(), '<f8')
Rp, R, body_h = np.frombuffer(raw[56:68].tobytes(), '<f4')
fb, = np.frombuffer(raw[156:160].tobytes(), '<u4')          # 末尾の在荷フラグ [byte]

body = raw[int(hb):].reshape(-1, int(rb))                   # レコード単位のバイト列
nf = int(rb) - int(fb)                                      # 浮動小数のぶん
el = '<f4' if eb == 4 else '<f8'
a = np.frombuffer(body[:, :nf].tobytes(), el).reshape(len(body), nf // int(eb))
(t, psi, th_t, omega, alpha, th_m, j_load, torque_table, torque_input,
 torque_slosh, m_bend, m_bend_x, m_bend_y, f_tab_x, f_tab_y) = a[:, :int(nsc)].T
st = a[:, int(nsc):].reshape(len(a), int(nst), int(nps))    # [時刻, ステーション, 量]
V, h, phi_t, phi_r, spill = (st[:, :, k] for k in range(int(nps)))
present = body[:, nf:nf + int(nst)]                         # u8 の 0/1
```

`core/dumpio.py` がこの読み方をそのまま実装している（イベント列も）。

## 8. 突き合わせの型

`core/compare_ref.py` が、**同じ params.json・同じ初期状態**で

- C: `filler --f64` の出力
- Python: `py/ref.py` の `simulate()`

を回して、15 スカラ + ステーション毎 5 量を 1 レコードずつ比べる。
両者は同じ式・同じ積分順序・同じ刻みなので、差の出どころは演算順序の丸めだけになる。
しきい値とその根拠は `compare_ref.py` の先頭に書いてある。

前版のような「加速度だけを共通入力にして液まわりだけ比べる」形は取っていない。
rev.3 では割出し軸に制御ループが無く、テーブル角はカム曲線で一意に決まるので、
**軸も含めて丸ごと突き合わせられる。**
