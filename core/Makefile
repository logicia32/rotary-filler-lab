# ロータリー充填機 仮想モデル / 物理コア（params.json rev.3）
#
# -ffast-math は使わない。Python の参照実装（py/ref.py）と数値を突き合わせるので、
# 演算順序を変える最適化は入れない。x87 の 80bit 中間結果も同じ理由で避ける。

CC      ?= cc
CSTD    := -std=c99
WARN    := -Wall -Wextra -Wshadow -Wconversion -Wno-sign-conversion
OPT     := -O2
CFLAGS  := $(CSTD) $(WARN) $(OPT) -D_DEFAULT_SOURCE
LDLIBS  := -lm

TARGET  := filler
SHLIB   := libtwin.so
TESTBIN := test_twin

PY ?= ../.venv/bin/python

all: $(TARGET) $(SHLIB)

# 単体実行
$(TARGET): twin.o main.o
	$(CC) $(CFLAGS) -o $@ twin.o main.o $(LDLIBS)

twin.o: twin.c twin.h
	$(CC) $(CFLAGS) -c -o $@ twin.c

main.o: main.c twin.h
	$(CC) $(CFLAGS) -c -o $@ main.c

# 単体試験（期待値は py/ref.py の実測値）
$(TESTBIN): twin.o test_twin.o
	$(CC) $(CFLAGS) -o $@ twin.o test_twin.o $(LDLIBS)

test_twin.o: test_twin.c twin.h
	$(CC) $(CFLAGS) -c -o $@ test_twin.c

# Python から ctypes で呼ぶ用
$(SHLIB): twin.c twin.h
	$(CC) $(CFLAGS) -fPIC -shared -o $@ twin.c $(LDLIBS)

# 動作確認: 1 タクト回してダンプを作る
run: $(TARGET)
	./$(TARGET) --cycles 1 --out out.bin

# 正常時と故障 4 種を回す。センサ合成側の入力一式。
DUR ?= 30.0
dataset: $(TARGET)
	./$(TARGET) --duration $(DUR) --out normal.bin
	./$(TARGET) --duration $(DUR) --fault-missing --out fault_missing.bin
	./$(TARGET) --duration $(DUR) --fault-valve   --out fault_valve.bin
	./$(TARGET) --duration $(DUR) --fault-cam     --out fault_cam.bin
	./$(TARGET) --duration $(DUR) --fault-bearing --out fault_bearing.bin

# ---------------------------------------------------------------
# 試験
#   unit    : C 単体（諸元から出る量が期待値と合うか）
#   compare : C と py/ref.py の突き合わせ（同じ params・同じ初期状態）
#   events  : イベント列が取り決めどおりか
#   steady  : 排出と供給で定常状態になるか
# ---------------------------------------------------------------
CMP_CYCLES ?= 2

unit: $(TESTBIN)
	./$(TESTBIN)

# 正常時と、故障を全部入れた場合の 2 通りで突き合わせる。
# 故障側は参照実装にも同じ faults を有効にして読ませる（compare_ref.py --faults）。
compare: $(TARGET)
	@$(PY) -c "import numpy" 2>/dev/null || \
	  { echo "numpy が要ります: PY=<python> make compare"; exit 1; }
	./$(TARGET) --cycles $(CMP_CYCLES) --f64 --out .cmp.bin --quiet
	$(PY) compare_ref.py .cmp.bin --cycles $(CMP_CYCLES)
	./$(TARGET) --cycles $(CMP_CYCLES) --f64 --out .cmpf.bin --quiet \
	            --fault-missing --fault-valve --fault-cam --fault-bearing
	$(PY) compare_ref.py .cmpf.bin --cycles $(CMP_CYCLES) --faults

# 衝撃系（カムフォロワ・軸受）だけを有効にして、連続量が正常時と 1 ビットも
# 変わらないことまで見る。そのあと受け渡し系（欠品・弁）を別に見る。
events: $(TARGET)
	./$(TARGET) --cycles 4 --fault-bearing --fault-cam --out .fault.bin --quiet
	$(PY) check_events.py .fault.bin .fault.bin.events
	./$(TARGET) --cycles 12 --fault-missing --fault-valve --out .fault2.bin --quiet
	$(PY) check_events.py .fault2.bin .fault2.bin.events

steady: $(TARGET)
	./$(TARGET) --cycles 26 --log-rate 200 --out .steady.bin --quiet
	$(PY) check_steady.py .steady.bin

test: unit compare events steady

clean:
	rm -f $(TARGET) $(SHLIB) $(TESTBIN) twin.o main.o test_twin.o \
	      out.bin out.bin.events \
	      normal.bin fault_missing.bin fault_valve.bin fault_cam.bin fault_bearing.bin \
	      normal.bin.events fault_missing.bin.events fault_valve.bin.events \
	      fault_cam.bin.events fault_bearing.bin.events \
	      .cmp.bin .cmp.bin.events .cmpf.bin .cmpf.bin.events \
	      .steady.bin .steady.bin.events \
	      .fault.bin .fault.bin.events .fault2.bin .fault2.bin.events .noimp.bin \
	      trace.bin

.PHONY: all run dataset unit compare events steady test clean
