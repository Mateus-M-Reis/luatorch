.PHONY: all build test bench clean repl

all: build

build:
	./scripts/build.sh

test: build
	./scripts/test.sh

bench: build
	./scripts/bench.sh

repl: build
	./scripts/repl.sh

clean:
	rm -rf build
