CC = g++
MPICC = mpicxx

default:
	@$(CC) -O3 -DTIME -std=c++14 main.cc viterbi_serial.cc -o serial
	@$(MPICC) -O3 -fopenmp -DTIME -std=c++14 \
	    main.cc viterbi_parallel.cc -o parallel

debug:
	@$(CC) -O3 -std=c++14 -DDEBUG main.cc viterbi_serial.cc -o serial
	@$(MPICC) -O3 -fopenmp -std=c++14 \
	    -DDEBUG main.cc viterbi_parallel.cc -o parallel

run:
	@./serial $(seq)
	@mpirun -n $(n) parallel $(seq)

time:
	@time -p ./serial $(seq)
	@time -p mpirun -n $(n) parallel $(seq)

mkdata:
	@$(CC) -O3 -std=c++14 datagen.cc -o datagen
	@rm -rf data
	@mkdir data
	@./datagen

clean:
	@rm -rf datagen parallel serial *.dSYM output*
