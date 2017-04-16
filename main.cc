//
//  main.cc
//  Viterbi
//
//  Created by Shubham Chaudhary on 19/03/17.
//  Copyright © 2017 Shubham Chaudhary. All rights reserved.
//

#include <iostream>
#include <chrono>

#include "viterbi.hh"

int main(int argc, const char * argv[]) {
    std::string arg1 = std::string(argc > 1 ? argv[1] : "32");
    std::string arg2 = std::string(argc > 2 ? argv[2] : "64");

#ifdef DEBUG
    std::cout << "Running with hidden states "
               + arg1
               + " and sequence length "
               + arg2
               + "." << std::endl;
#endif

    // Initialize the algorithm
    viterbi instance(arg1, arg2);

    // Run the algorithm
    instance.decode();

#ifdef DEBUG
    // Print the predicted sequence
    instance.show_predicted();
#endif
    return 0;
}
