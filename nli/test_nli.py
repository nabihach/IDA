from nli_model import *
import nli_preprocessor
import sys

def load_model(file_name, test_genre):
    model_path = MODEL_DIR + file_name
    model, initial_epoch = BiLSTMModel().import_state(model_path, load_epoch=True)
    model_dir=os.path.dirname(model_path)+"/"

    test_set = nli_preprocessor.genre_test_set(test_genre)
    #test_set = nli_preprocessor.dev_matched_text_hyp_labels(genre=test_genre)
    #test_set = nli_preprocessor.get_multinli_matched_val_set()
    #test_set = nli_preprocessor.dev_mismatched_text_hyp_labels(genre=test_genre)
    #test_set = nli_preprocessor.get_multinli_mismatched_val_set()
    #test_set = nli_preprocessor.get_multinli_text_hyp_labels(genre=test_genre)
    # Tokenize sentences
    test_set["text"] = nli_preprocessor.tokenize_sentences(test_set["text"], model.wd)
    test_set["hyp"] = nli_preprocessor.tokenize_sentences(test_set["hyp"], model.wd)
    print(model.score(test_set))


if __name__ == '__main__':
    model_file = sys.argv[1]
    genre = sys.argv[2]
    load_model(model_file, genre)
