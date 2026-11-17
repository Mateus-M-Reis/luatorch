.PHONY: all build test bench compare clean repl install

all: build

build:
	./scripts/build.sh

test: build
	./scripts/test.sh

bench: build
	./scripts/bench.sh

compare: build
	./scripts/compare.sh

repl: build
	./scripts/repl.sh

clean:
	rm -rf build
