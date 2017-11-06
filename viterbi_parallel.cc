//
//  viterbi.cc
//  Viterbi
//
//  Created by Shubham Chaudhary on 19/03/17.
//  Copyright © 2017 Shubham Chaudhary. All rights resized.
//

#include <iostream>
#include <limits>
#include <vector>
#include <random>
#include <fstream>
#include <utility>
#include <exception>

#include <mpi.h>
#include <sys/mman.h>

#include "viterbi.hh"

int num_procs, tid, lp, rp;

#pragma omp declare reduction(                                           \
        max_argmax                                                       \
        : std::pair<double, int>                                           \
        : omp_out = (omp_in.first > omp_out.first ? omp_out : omp_in))     \
initializer (omp_priv =                                          \
        std::make_pair(-std::numeric_limits<double>::max(),      \
            std::numeric_limits<int>::max()))

template <typename T>
inline bool is_parallel(std::vector<T>& a, std::vector<T>& b) {
    bool parallel = a.size() == b.size();
    int length = std::min(a.size(), b.size());

    for (int i = 0; i < length - 1; i++) {
        if (!parallel)
            break;

        if (a[i] - b[i] != a[i + 1] - b[i + 1])
            parallel = false;
    }

    return parallel;
}

viterbi::viterbi(std::string &arg1, std::string& arg2) {
    // Initialise MPI runtime information
    MPI_Init (nullptr, nullptr);

    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &tid);

    // Load info of the decoder
    std::FILE *file_p = std::fopen(("data/viterbi_" + arg1 + "_" + arg2).c_str(), "r");

    // Calculate file size
    std::fseek(file_p, 0, SEEK_END);
    unsigned length = std::ftell(file_p);

    // Memory map the file
    char *file_map = (char*) mmap(
            nullptr,
            length,
            PROT_READ,
            MAP_PRIVATE,
            fileno(file_p),
            0);

    // Get sequence info
    seq_len = *((unsigned int*)(file_map) + 0);
    num_hidden = *((unsigned int*)(file_map) + 1);

    // Determine the range of processing
    lp = (seq_len / num_procs) * tid - 1;
    rp = (seq_len / num_procs) * (tid + 1) - 1;

    if (tid == num_procs - 1)
        rp = seq_len - 1;

    unsigned int offset = sizeof(unsigned int) * 2
        + sizeof(double) * (lp + 1) * num_hidden * num_hidden;

    // Allocate memory for data
    ltdp_matrix.resize(rp - lp);

    for (auto& a : ltdp_matrix) {
        a.resize(num_hidden);

        for (auto& b: a) {
            b.resize(num_hidden);
        }
    }

    double *matrix = (double*) (file_map + offset);

    // Load LTDP matrix
    for (int i = 0; i < rp - lp; i++) {

#ifdef WAVEFRONT
#pragma omp parallel for default(shared)
#endif
        for (int j = 0; j < num_hidden; j++) {
            for (int k = 0; k < num_hidden; k++) {
                ltdp_matrix[i][j][k] =
                    matrix[k + num_hidden * j + num_hidden * num_hidden * i];
            }
        }
    }

    std::fclose(file_p);

    // Allocate memory for sequence
    predicted_sequence.resize(rp - lp);

#ifdef DEBUG
    std::cout << "Finished loading data from disk." << std::endl;
#endif
}

void viterbi::decode() {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> dist_rand(-100.0, 0);

    // Setup dynamic programming tables
    std::vector<std::vector<double>> dp1;
    std::vector<std::vector<unsigned int>> dp2;

    // Test variables
    uint8_t conv, global_conv;

    // Allocate memory for the dp tables
    dp1.resize(rp - lp);
    dp2.resize(rp - lp);

    for (auto& a : dp1)
        a.resize(num_hidden);

    for (auto& a : dp2)
        a.resize(num_hidden);

#ifdef DEBUG
    std::cout << "Running algorithm for " << num_hidden
        << " hidden states, and a sequence of length "
        << seq_len << std::endl;
#endif

    /* BEGIN LTDP PARALLEL VITERBI ALGORITHM
    */
    // Time the execution
    double start, end;

    if (tid == 0)
        start = MPI_Wtime();


#ifdef DEBUG
    std::cout << "Forward pass: Thread "
        + std::to_string(tid)
        + "/"
        + std::to_string(num_procs)
        + " running from "
        + std::to_string(lp + 1)
        + " to "
        + std::to_string(rp)
        + "\n";
#endif

    // Initialize local vector
    std::vector<double> s(num_hidden, 0.0);

    if (tid != 0) {
        for (auto& a : s) {
            a = dist_rand(generator);
        }
    }

    for (int i = lp + 1; i <= rp; i++) {

#ifdef WAVEFRONT
#pragma omp parallel for default(shared)
#endif
        for (int j = 0; j < num_hidden; j++) {
            std::pair<double, int> max_p = std::make_pair(
                    -std::numeric_limits<double>::max(),
                    std::numeric_limits<int>::max());
            double entry;

            for (size_t k = 0; k < num_hidden; k++) {
                entry = s[k] + ltdp_matrix[i - lp - 1][j][k];

                if (entry > max_p.first) {
                    max_p.first = entry;
                    max_p.second = k;
                }
            }

            dp1[i - lp - 1][j] = max_p.first;
            dp2[i - lp - 1][j] = max_p.second;
        }

        s = dp1[i - lp - 1];
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Fix up loop
    do {
        conv = 0b1;

        // Initialize local vector
        std::vector<double> s(num_hidden, 0.0), next_s(num_hidden, 0.0);

#ifdef DEBUG
        std::cout << "Forward fix up loop: Thread "
            + std::to_string(tid)
            + " running from "
            + std::to_string(lp + 1)
            + " to "
            + std::to_string(rp)
            + "\n";
#endif

        if (tid != num_procs - 1) {
            // Send the solution vector to the next section
            MPI_Send(
                    &dp1[rp - lp - 1].front(),
                    num_hidden,
                    MPI_DOUBLE,
                    tid + 1,
                    0,
                    MPI_COMM_WORLD);
        }

        if (tid != 0) {
            conv = 0b0;

            // Get the solution vector of the previous section
            MPI_Recv(
                    &s.front(),
                    num_hidden,
                    MPI_DOUBLE,
                    tid - 1,
                    0,
                    MPI_COMM_WORLD,
                    nullptr);
        }

#ifdef DEBUG
        std::cout << "Forward fix up loop: Thread "
            + std::to_string(tid)
            + "/"
            + std::to_string(num_procs)+
            + " finished communication.\n";
#endif

        if (tid != 0) {
            for (int i = lp + 1; i <= rp; i++) {

#ifdef WAVEFRONT
#pragma omp parallel for default(shared)
#endif
                for (int j = 0; j < num_hidden; j++) {
                    std::pair<double, int> max_p = std::make_pair(
                            -std::numeric_limits<double>::max(),
                            std::numeric_limits<int>::max());
                    double entry;

                    for (size_t k = 0; k < num_hidden; k++) {
                        entry = s[k] + ltdp_matrix[i - lp - 1][j][k];

                        if (entry > max_p.first) {
                            max_p.first = entry;
                            max_p.second = k;
                        }
                    }

                    next_s[j] = max_p.first;

                    dp2[i - lp - 1][j] = max_p.second;
                }

                s = next_s;

                if (is_parallel(s, dp1[i - lp - 1])) {
                    conv = 0b1;
                    break;
                }

                dp1[i - lp - 1] = s;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

#ifdef DEBUG
        std::cout << "Forward fix up loop: Thread "
            + std::to_string(tid)
            + " finished forward pass.\n";
#endif

        // Get the global value of conv
        MPI_Allreduce(&conv, &global_conv, 1, MPI_BYTE, MPI_LAND, MPI_COMM_WORLD);

#ifdef DEBUG
        std::cout << "Forward fix up loop: Thread "
            + std::to_string(tid)
            + " sent value of conv as "
            + (conv ? "true" : "false")
            + " and recieved global conv as "
            + (global_conv ? "true" : "false")
            + "\n";
#endif

    } while (!global_conv);

    // Backward pass

#ifdef DEBUG
    std::cout << "Backward pass: Thread "
        + std::to_string(tid)
        + " running from "
        + std::to_string(rp)
        + " to "
        + std::to_string(lp + 1)
        + "\n";
#endif

    // Initialize local variable
    unsigned int x = 0;

    for (int i = rp; i >= lp + 1; i--) {
        predicted_sequence[i - lp - 1] = dp2[i - lp - 1][x];
        x = predicted_sequence[i - lp - 1];
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Fix up loop for backward pass
    do {
        conv = 0b1;

        // Initialize local variable
        unsigned int x;

#ifdef DEBUG
        std::cout << "Backward fix up loop: Thread "
            + std::to_string(tid)
            + " running from "
            + std::to_string(rp)
            + " to "
            + std::to_string(lp + 1)
            + "\n";
#endif

        if (tid != 0) {
            // Send the result value to the previous section
            MPI_Send(
                    &predicted_sequence[0],
                    1,
                    MPI_UNSIGNED,
                    tid - 1,
                    0,
                    MPI_COMM_WORLD);
        }
        if (tid != num_procs - 1) {
            conv = 0b0;

            // Get the result value of the next section
            MPI_Recv(
                    &x,
                    1,
                    MPI_UNSIGNED,
                    tid + 1,
                    0,
                    MPI_COMM_WORLD,
                    nullptr);

        }

#ifdef DEBUG
        std::cout << "Backward fix up loop: Thread "
            + std::to_string(tid)
            + "/"
            + std::to_string(num_procs)+
            + " finished communication.\n";
#endif

        if (tid != num_procs - 1) {
            for (int i = rp; i >= lp + 1; i--) {
                x = dp2[i - lp - 1][x];

                if (predicted_sequence[i - lp - 1] == x) {
                    conv = 0b1;
                    break;
                }

                predicted_sequence[i - lp - 1] = x;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

#ifdef DEBUG
        std::cout << "Backward fix up loop: Thread "
            + std::to_string(tid)
            + " finished backward pass.\n";
#endif

        // Get the global value of conv
        MPI_Allreduce(&conv, &global_conv, 1, MPI_BYTE, MPI_LAND, MPI_COMM_WORLD);

#ifdef DEBUG
        std::cout << "Backward fix up loop: Thread "
            + std::to_string(tid)
            + " sent value of conv as "
            + (conv ? "true" : "false")
            + " and recieved global conv as "
            + (global_conv ? "true" : "false")
            + "\n";
#endif

    } while (!global_conv);

    MPI_Barrier(MPI_COMM_WORLD);

    if (tid == 0) {
        end = MPI_Wtime();
        std::cout << "Total time elapsed: "
            << end - start
            << std::endl;
    }

    MPI_Finalize();
}

void viterbi::show_predicted() {
    std::ofstream file(
            "output_parallel_" + std::to_string(tid) + "_" + std::to_string(num_procs),
            std::ios::out | std::ios::trunc);

    for (auto& a : predicted_sequence)
        file << a << " ";
    file << std::endl;

    file.close();
}
