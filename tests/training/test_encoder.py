from torch import tensor
from transformers import EncoderDecoderModel, NllbTokenizer, T5EncoderModel, T5Config
# from transformers import EncoderDe
from transformers.adapters import XLMRobertaAdapterModel , AdapterConfig

src = tensor([[[   36],
         [   26],
         [  882],
         [   42],
         [ 8611],
         [   95],
         [  107],
         [    4],
         [ 1554],
         [   16],
         [ 5684],
         [  246],
         [   40],
         [ 2627]],

        [[ 1053],
         [    4],
         [ 5985],
         [ 2833],
         [   20],
         [15528],
         [   31],
         [15529],
         [ 2833],
         [    4],
         [ 1554],
         [ 3622],
         [    4],
         [  122]],

        [[ 1554],
         [   25],
         [   49],
         [24475],
         [    5],
         [ 2013],
         [   49],
         [ 6037],
         [24476],
         [  174],
         [   14],
         [  111],
         [    9],
         [  214]],

        [[    4],
         [ 1533],
         [   92],
         [10611],
         [ 1894],
         [    4],
         [  389],
         [   27],
         [   21],
         [10538],
         [    4],
         [  251],
         [   71],
         [   94]],

        [[    4],
         [  254],
         [  796],
         [24487],
         [ 2694],
         [   21],
         [ 1094],
         [10612],
         [    6],
         [ 2421],
         [   15],
         [    5],
         [   15],
         [ 1492]],

        [[15546],
         [   39],
         [  191],
         [  116],
         [    5],
         [  422],
         [   29],
         [   28],
         [  570],
         [ 8613],
         [ 8612],
         [   75],
         [   11],
         [  183]],

        [[    9],
         [   73],
         [  630],
         [   26],
         [   15],
         [    7],
         [  619],
         [24491],
         [   10],
         [   83],
         [  286],
         [   12],
         [ 1362],
         [   59]],

        [[  238],
         [    4],
         [  770],
         [    8],
         [ 3475],
         [   10],
         [  748],
         [   12],
         [  117],
         [    8],
         [    4],
         [ 5992],
         [   91],
         [15562]],

        [[ 5991],
         [    6],
         [  154],
         [ 4130],
         [   26],
         [ 1124],
         [    9],
         [    4],
         [  435],
         [   27],
         [   53],
         [10614],
         [ 2048],
         [   24]],

        [[  123],
         [   13],
         [  154],
         [   59],
         [   15],
         [  138],
         [ 2675],
         [10615],
         [    9],
         [   49],
         [24514],
         [  844],
         [    4],
         [ 4585]],

        [[    8],
         [15571],
         [  844],
         [ 3057],
         [   12],
         [   53],
         [   97],
         [    9],
         [  456],
         [   15],
         [  397],
         [    7],
         [ 4545],
         [  232]],

        [[ 4253],
         [  157],
         [   93],
         [   48],
         [  117],
         [  281],
         [ 2176],
         [   12],
         [ 1899],
         [   15],
         [15576],
         [   48],
         [ 2161],
         [    4]],

        [[24518],
         [   51],
         [  196],
         [  477],
         [   24],
         [24519],
         [   74],
         [   23],
         [   19],
         [  230],
         [   93],
         [   84],
         [   52],
         [   16]],

        [[  245],
         [  743],
         [   40],
         [   57],
         [15580],
         [  174],
         [   10],
         [  660],
         [   16],
         [ 8197],
         [    8],
         [  490],
         [ 3671],
         [15581]],

        [[   12],
         [   74],
         [   15],
         [   57],
         [  187],
         [    5],
         [10518],
         [   91],
         [   59],
         [   43],
         [   84],
         [10119],
         [  174],
         [   10]],

        [[15584],
         [  196],
         [    5],
         [    7],
         [ 1910],
         [ 8623],
         [15585],
         [  694],
         [   36],
         [  244],
         [  286],
         [   12],
         [   36],
         [   11]]])

# model = XLMRobertaAdapterModel.from_pretrained("xlm-roberta-base")

#enc_dec_mode = EncoderDecoderModel.from_pretrained("facebook/nllb-200-distilled-600M", "xlm-roberta-base", "xlm-roberta-base")
# enc_dec_model_pre = EncoderDecoderModel.from_encoder_decoder_pretrained("facebook/nllb-200-distilled-600M", "xlm-roberta-base")

# tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

# t5_model = T5EncoderModel.from_pretrained("t5-small")

# print(model(src))

# print(t5_model(src))

decoder = XLMRobertaAdapterModel.from_pretrained("xlm-roberta-base")
adapter_config = AdapterConfig.load("pfeiffer")
decoder.add_adapter("posdep_english_adapter_decoder",adapter_config)
decoder.train()
decoder.set_active_adapters("posdep_english_adapter_decoder")

print(decoder(src))
print(decoder(src).last_hidden_state)