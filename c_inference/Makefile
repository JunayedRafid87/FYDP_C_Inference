CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -pthread -fPIC -Iinclude
LDFLAGS ?= -lm -lpthread

SRCS = src/postprocess.c src/bpu_yolo_infer.c src/v4l2_capture.c
OBJS = $(SRCS:.c=.o)

all: build/libpostprocess.so build/bpu_infer_cli python_binding/libpostprocess.so

build:
	mkdir -p build

src/%.o: src/%.c | build
	$(CC) $(CFLAGS) -c $< -o $@

build/libpostprocess.so: src/postprocess.o | build
	$(CC) -shared -o $@ $^ $(LDFLAGS)

python_binding/libpostprocess.so: build/libpostprocess.so
	cp build/libpostprocess.so python_binding/libpostprocess.so

build/bpu_infer_cli: src/main_infer.o $(OBJS) | build
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf build src/*.o python_binding/libpostprocess.so

.PHONY: all clean
