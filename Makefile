run:
	cargo run

preview: run
	dot -Tsvg outputs/graph.dot -o outputs/graph.svg
	open outputs/graph.svg