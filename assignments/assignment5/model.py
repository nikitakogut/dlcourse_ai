import torch
import torch.nn as nn

class Word2VecNegativeSamples(nn.Module):
    def __init__(self, num_tokens, wordvec_dim):
        super(Word2VecNegativeSamples, self).__init__()
        self.input = nn.Linear(num_tokens, wordvec_dim, bias=False)
        self.output = nn.Linear(wordvec_dim, num_tokens, bias=False)
        self.wordvec_dim = wordvec_dim
        
    def forward(self, input_index_batch, output_indices_batch):
        '''
        Implements forward pass with negative sampling
        
        Arguments:
        input_index_batch - Tensor of ints, shape: (batch_size, ), indices of input words in the batch
        output_indices_batch - Tensor if ints, shape: (batch_size, num_negative_samples+1),
                                indices of the target words for every sample
                                
        Returns:
        predictions - Tensor of floats, shape: (batch_size, um_negative_samples+1)
        '''
        
        '''
        Гипотеза:
        Основная проблема - подсчет векторов. Лучше это делать одним действием,
        чтобы градиенты считались по одному тензору
        
        -----------------------------------
        for i in range(len(input_index_batch)):
            input_vector = self.input.weight[:, input_index_batch[i]]
            output_vector = self.output.weight[output_indices_batch[i]].t()
            results[i] = input_vector.matmul(output_vector)
        return results
        -----------------------------------
        cpu: 2:00 / epoch
        gpu: 0:58 / epoch
        -----------------------------------
        
        -----------------------------------
        a = self.input.weight[:, input_index_batch].t().reshape((len(input_index_batch), 1, wordvec_dim))
        b = torch.zeros((len(output_indices_batch), wordvec_dim, num_negative_samples+1))
                 #.type(torch.cuda.FloatTensor)
        for i in range(len(output_indices_batch)):
            b[i] = self.output.weight[output_indices_batch[i]].t()
        c = a.matmul(b).reshape(len(output_indices_batch), num_negative_samples+1)
        return c
        -----------------------------------
        cpu: 0:50 / epoch
        gpu: 0:42 / epoch
        -----------------------------------
        
        -----------------------------------
        a = self.input.weight[:, input_index_batch].t()\
            .reshape((len(input_index_batch), 1, wordvec_dim))
        d = torch.transpose(self.output.weight[output_indices_batch], 1, 2)
        c = a.matmul(d).reshape(len(output_indices_batch), num_negative_samples+1)
        return c
        -----------------------------------
        cpu: 0:22 / epoch
        gpu: 0:09 / epoch
        + loss убывает намного лучше (скорее всего меньше погрешность)
        -----------------------------------
        
        '''    
        batch_size = output_indices_batch.shape[0]
        num_samples = output_indices_batch.shape[1]
        
        embedding_vectors = self.input.weight[:, input_index_batch].t()\
                                .reshape((batch_size, 1, self.wordvec_dim))
        context_vectors = torch.transpose(self.output.weight[output_indices_batch], 1, 2)
        return embedding_vectors.matmul(context_vectors)\
               .reshape(batch_size, num_samples)
