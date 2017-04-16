//
//  viterbi.h
//  Viterbi
//
//  Created by Shubham Chaudhary on 19/03/17.
//  Copyright © 2017 Shubham Chaudhary. All rights resized.
//

#ifndef viterbi_h
#define viterbi_h

#include <vector>
#include <iostream>
#include <fstream>

/* A struct to hold the data of the viterbi decoder.
 *
 *
 *  - The structure holds the following data items:
 *  - The observation space O of size N.
 *  - The state space S of size K.
 *  - An array of initial probabilities, P of size K.
 *  - A sequence of observations Y of size T.
 *  - Transition matrix A of size K x K.
 *  - Emission matrix of size K x N.
 *
 *  - The most likely sequence X of size T.
 */
class viterbi {
private:
    unsigned int num_hidden, seq_len;
    std::vector<unsigned int> predicted_sequence;
    std::vector<std::vector<std::vector<double>>> ltdp_matrix;

public:
    viterbi(std::string&, std::string&);

    void decode();

    void show_predicted();
};


#endif /* viterbi_h */
