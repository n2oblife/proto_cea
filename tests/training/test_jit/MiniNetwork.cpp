
#include <iostream>
#include <vector>

// #include <torch/nn.h>
// #include <torch/script.h>
#include <ctranslate2/models/model.h>
#include "ctranslate2/translator.h"

namespace ct2 = ctranslate2;


int main(int argc, char* argv[]) {

    std::string path("/home/zk274707/Projet/proto_utils/save_dir/test_jit/test.pt"); 
    std::string model_path("opus-mt-en-de");
    const auto model = ct2::models::Model::load(path);
    const ct2::models::ModelLoader model_loader(path);
    ct2::Translator translator(model_loader);

    // std::vector<std::vector<std::string>> batch = {{"▁Hello", "▁World", "!", "</s>"}};
    // auto translation = translator.translate_batch(batch);
    // for (auto &token:  translation[0].output()) {
    //     std::cout << token << ' ';
    // }
    std::cout << std::endl;
}