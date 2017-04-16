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
#include <chrono>

#include "viterbi.hh"

viterbi::viterbi(std::string &arg1, std::string& arg2) {
    std::ifstream file;

    // Load info of the decoder
    file.open("data/viterbi_" + arg1 + "_" + arg2, std::ios::in | std::ios::binary);
    file.read((char*) &seq_len, sizeof(seq_len));
    file.read((char*) &num_hidden, sizeof(num_hidden));

    // Allocate memory for data
    ltdp_matrix.resize(seq_len);

    for (auto& a : ltdp_matrix) {
        a.resize(num_hidden);

        for (auto& b: a) {
            b.resize(num_hidden);
        }
    }

    // Load LTDP matrix
    for (auto& row : ltdp_matrix)
        for (auto& col : row)
            for (auto& e : col)
                file.read((char*) &e, sizeof(e));

    file.close();

    // Allocate memory for sequence
    predicted_sequence.resize(seq_len);

#ifdef DEBUG
    std::cout << "Finished loading data from disk." << std::endl;
#endif
}

void viterbi::decode() {
    // Setup dynamic programming tables
    std::vector<std::vector<double>> dp1;
    std::vector<std::vector<unsigned int>> dp2;

    // Allocate memory for the dp tables
    dp1.resize(seq_len);
    dp2.resize(seq_len);

    for (auto& a : dp1)
        a.resize(num_hidden);

    for (auto& a : dp2)
        a.resize(num_hidden);

#ifdef DEBUG
    std::cout << "Running algorithm for " << num_hidden
              << " hidden states, and a sequence of length "
              << seq_len << std::endl;
#endif

    // Algorithm
    std::vector<double> s(num_hidden, 0.0);

    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < num_hidden; j++) {
            double max = -std::numeric_limits<double>::max();
            size_t argmax = std::numeric_limits<size_t>::max();

            double entry;
            for (size_t k = 0; k < num_hidden; k++) {
                entry = s[k] + ltdp_matrix[i][j][k];

                if (entry > max) {
                    max = entry;
                    argmax = k;
                }
            }

            dp1[i][j] = max;
            dp2[i][j] = argmax;
        }

        s = dp1[i];
    }

    unsigned int x = 0;

    for (int i = seq_len - 1; i >= 0; i--) {
        predicted_sequence[i] = dp2[i][x];
        x = predicted_sequence[i];
    }
}

void viterbi::show_predicted() {
    std::ofstream file(
            "output_serial",
            std::ios::out | std::ios::trunc);

    for (auto& a : predicted_sequence)
        file << a << " ";
    file << std::endl;

    file.close();
}
