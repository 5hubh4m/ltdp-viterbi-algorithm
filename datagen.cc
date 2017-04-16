#include <iostream>
#include <fstream>
#include <cmath>
#include <valarray>
#include <random>

int main() {
    int num_observable = 256;

    std::default_random_engine generator;
    std::uniform_real_distribution<double> dist_rand(0.5, 1.0);
    std::uniform_int_distribution<int> dist_obs(0, num_observable - 1);

    std::ofstream file;

    for (int num_hidden = 32; num_hidden <=2048 ; num_hidden *= 2) {
        std::valarray<std::valarray<double>> transistion(
                std::valarray<double>(0.0f, num_hidden), num_hidden);
        std::valarray<std::valarray<double>> emission(
                std::valarray<double>(0.0f, num_observable), num_hidden);
        std::valarray<double> prior(0.0f, num_hidden);

        for (auto& row : transistion)
            for (auto& num : row)
                num = dist_rand(generator);

        for (auto& row : emission)
            for (auto& num : row)
                num = dist_rand(generator);

        for (auto& num : prior)
            num = dist_rand(generator);

        prior /= prior.sum();

        for (auto& row : transistion)
            row /= row.sum();
        for (auto& row : emission)
            row /= row.sum();

        prior = std::log(prior);
        transistion = std::log(transistion);
        emission = std::log(emission);


        for (int seq_len = 64; seq_len <= 1024; seq_len *= 2) {
            std::valarray<unsigned int> sequence((unsigned int) 0, seq_len - 1);

            for (auto& num : sequence)
                num = dist_obs(generator);

            file.open("data/viterbi_"
                    + std::to_string(num_hidden)
                    + "_"
                    + std::to_string(seq_len),
                std::ios::binary | std::ios::out | std::ios::trunc);

            file.write((char*) &seq_len, sizeof(seq_len));
            file.write((char*) &num_hidden, sizeof(num_hidden));

            double e;

            for (unsigned int i = 0; i < seq_len; i++) {
                for (unsigned int j = 0; j < num_hidden; j++) {
                    for (unsigned int k = 0; k < num_hidden; k++) {

                        if (i == 0) {
                            e = prior[j] + emission[j][sequence[0]];
                        }
                        else if (i == seq_len - 1) {
                            e = 0;
                        }
                        else {
                            e = transistion[k][j] + emission[j][sequence[i]];
                        }

                        file.write((char*) &e, sizeof(e));
                    }
                }
            }

            std::cout << "Finished loading matrix with "
                      << num_hidden
                      << " hidden states and "
                      << seq_len
                      << " number of observations."
                      << std::endl;

            file.close();
        }
    }

    return 0;
}
