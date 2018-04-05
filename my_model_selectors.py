import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

    def get_components(self):
        """
        generate components between min_n_components and max_n_components (included).
        """
        n_components = np.arange(self.min_n_components, self.max_n_components+1)
        for n_component in n_components:
            yield n_component

class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        models = []
        
        # log of sample size
        logN = np.log(len(self.X))
        # number of features
        n_features = self.X.shape[1]
        
        for n_component in self.get_components():
            hmm_model = self.base_model(n_component)
            
            # if failed to train, just ignore
            if not hmm_model:
                models.append((hmm_model, np.inf))
                continue

            try:
                # scoring likelihood
                logL = hmm_model.score(self.X, self.lengths)
                
                # estimating number of parameters
                # https://discussions.udacity.com/t/verifing-bic-calculation/246165/2
                p = n_component**2 + n_component*n_features*2 - 1
                
                # calculating BIC score
                BICscore = -2 * logL + p*logN

                models.append((hmm_model, BICscore))
            except:
                # if failed to score, just ignore
                models.append((hmm_model, np.inf))

        return min(models, key=lambda model:model[1])[0]

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        models = []
        
        # retrieving all words except the current one
        rest_words = [word for word, _ in self.words.items() if word != self.this_word]

        for n_component in self.get_components():
            hmm_model = self.base_model(n_component)
            
            # if failed to train, just ignore
            if not hmm_model:
                models.append((hmm_model, -np.inf))
                continue
            
            try:
                # scoring evidence likelihood
                logL = hmm_model.score(self.X, self.lengths)
                
                # scoring anti-evidence likelihood
                log_antiL = [hmm_model.score(self.hwords[word][0], self.hwords[word][1]) for word in rest_words]
            except:
                # if failed to score, just ignore
                models.append((hmm_model, -np.inf))
                continue
            
            # difference between evidence likelihood and the average of the anti-evidence likelihood
            DICscore = logL - sum(log_antiL)/len(rest_words)
            
            models.append((hmm_model, DICscore))
        
        return max(models, key=lambda model : model[1])[0]

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        models = []
        
        # defining split method
        if len(self.sequences) == 1:
            return None
        
        splits = 2 if len(self.sequences) <= 3 else 3
        split_method = KFold(n_splits=splits)
        
        for n_component in self.get_components():
            hmm_model = GaussianHMM(n_components=n_component, covariance_type="diag", n_iter=1000, 
                              random_state=self.random_state, verbose=False)

            likelihood = []
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                # splitting training and testing samples
                X_train, length_train = combine_sequences(cv_train_idx, self.sequences)
                X_test,  length_test  = combine_sequences(cv_test_idx,  self.sequences)
                
                try:
                    # fitting the traning samples
                    model = hmm_model.fit(X_train, length_train)
                    
                    # validating with testing samples
                    likelihood.append(model.score(X_test, length_test))
                except:
                    # if failed, just ignore
                    likelihood.append(-np.inf)
            
            # storing the average likelihood
            models.append((hmm_model, np.mean(likelihood)))
        return max(models, key=lambda model : model[1])[0]
