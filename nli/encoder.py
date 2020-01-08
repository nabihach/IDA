from utils import *

class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.0, base_rnn=nn.LSTM, pretrained_embeddings=None, dropout_p=0.1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size #vocabulary size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.base_rnn = base_rnn
        self.dropout_p = dropout_p
        self.num_mem_slots = 300
        self.mem_context_size = 300
        
        if pretrained_embeddings is not None:
            
            self.embedding = nn.Embedding(pretrained_embeddings.size(0), pretrained_embeddings.size(1))
            self.embedding.weight = nn.Parameter(pretrained_embeddings, requires_grad=True)
            print("GloVe embeddings added")
        else:
            self.embedding = nn.Embedding(input_size, hidden_size)

       
        self.dropout = nn.Dropout(dropout_p)
        self.forward_rnn = self.base_rnn(hidden_size+self.mem_context_size, hidden_size, n_layers)
        self.backward_rnn = self.base_rnn(hidden_size+self.mem_context_size, hidden_size, n_layers)

        self.target_mem_slots =  500
        self.target_mem_slots_2 = 500
        self.target_mem_slots_3 = 500
        self.target_mem_slots_4 = 500

        self.M_k_fwd = nn.Linear(hidden_size, self.num_mem_slots, bias=False)
        self.M_v_fwd = nn.Linear(self.num_mem_slots, self.mem_context_size, bias=False)
        self.M_k_bkwd = nn.Linear(hidden_size, self.num_mem_slots, bias=False)
        self.M_v_bkwd = nn.Linear(self.num_mem_slots, self.mem_context_size, bias=False)

        self.M_k_fwd_target = None# --> When testing on 1st target or training for 2nd target pad, use self.M_k_fwd_target =  nn.Linear(hidden_size, self.target_mem_slots, bias=False)
        self.M_v_fwd_target = None# --> When testing on 1st target or training for 2nd target pad, use self.M_v_fwd_target =  nn.Linear(self.target_mem_slots, self.mem_context_size, bias=False)
        self.M_k_bkwd_target = None# --> When testing on 1st target or training for 2nd target pad, use self.M_k_bkwd_target = nn.Linear(hidden_size, self.target_mem_slots, bias=False)
        self.M_v_bkwd_target = None# --> When testing on 1st target or training for 2nd target pad, use self.M_v_bkwd_target = nn.Linear(self.target_mem_slots, self.mem_context_size, bias=False)

        self.M_k_fwd_target_2 = None# --> When testing on 2nd target or training for 3rd target pad, use self.M_k_fwd_target_2 = nn.Linear(hidden_size, self.target_mem_slots, bias=False)
        self.M_v_fwd_target_2 = None#  --> When testing on 2nd target or training for 3rd target pad, use self.M_v_fwd_target_2 = nn.Linear(self.target_mem_slots, self.mem_context_size, bias=False)
        self.M_k_bkwd_target_2 = None# --> When testing on 2nd target or training for 3rd target pad, use self.M_k_bkwd_target_2 = nn.Linear(hidden_size, self.target_mem_slots, bias=False)
        self.M_v_bkwd_target_2 = None# --> When testing on 2nd target or training for 3rd target pad, use self.M_v_bkwd_target_2 = nn.Linear(self.target_mem_slots, self.mem_context_size, bias=False)

        self.M_k_fwd_target_3 = None# --> When testing on 3rd target or training for 4th target pad, use self.M_k_fwd_target_3 =  nn.Linear(hidden_size, self.target_mem_slots, bias=False)
        self.M_v_fwd_target_3 = None#  --> When testing on 3rd target or training for 4th target pad, use self.M_v_fwd_target_3 = nn.Linear(self.target_mem_slots, self.mem_context_size, bias=False)
        self.M_k_bkwd_target_3 = None# --> When testing on 3rd target or training for 4th target pad, use self.M_k_bkwd_target_3 =  nn.Linear(hidden_size, self.target_mem_slots, bias=False)
        self.M_v_bkwd_target_3 = None# --> When testing on 3rd target or training for 4th target pad, use self.M_v_bkwd_target_3 =  nn.Linear(self.target_mem_slots, self.mem_context_size, bias=False)

        self.M_k_fwd_target_4 = None# --> When testing on 4th target or training for 5th target pad, use self.M_k_fwd_target_4 =  nn.Linear(hidden_size, self.target_mem_slots, bias=False)
        self.M_v_fwd_target_4 = None#  --> When testing on 4th target or training for 5th target pad, use self.M_v_fwd_target_4 = nn.Linear(self.target_mem_slots, self.mem_context_size, bias=False)
        self.M_k_bkwd_target_4 = None# --> When testing on 4th target or training for 5th target pad, use self.M_k_bkwd_target_4 = nn.Linear(hidden_size, self.target_mem_slots, bias=False)
        self.M_v_bkwd_target_4 = None# --> When testing on 4th target or training for 5th target pad, use self.M_v_bkwd_target_4 = nn.Linear(self.target_mem_slots, self.mem_context_size, bias=False)


       
    def add_target_pad(self):
        self.M_k_fwd_target = nn.Linear(self.hidden_size, self.target_mem_slots, bias=False)
        self.M_v_fwd_target = nn.Linear(self.target_mem_slots, self.mem_context_size, bias=False)
        self.M_k_bkwd_target = nn.Linear(self.hidden_size, self.target_mem_slots, bias=False)
        self.M_v_bkwd_target = nn.Linear(self.target_mem_slots, self.mem_context_size, bias=False)

        if USE_CUDA:
            self.M_k_fwd_target = self.M_k_fwd_target.cuda()
            self.M_v_fwd_target = self.M_v_fwd_target.cuda()
            self.M_k_bkwd_target = self.M_k_bkwd_target.cuda()
            self.M_v_bkwd_target = self.M_v_bkwd_target.cuda()

    def add_target_pad_2(self):
        self.M_k_fwd_target_2 = nn.Linear(self.hidden_size, self.target_mem_slots_2, bias=False)
        self.M_v_fwd_target_2 = nn.Linear(self.target_mem_slots_2, self.mem_context_size, bias=False)
        self.M_k_bkwd_target_2 = nn.Linear(self.hidden_size, self.target_mem_slots_2, bias=False)
        self.M_v_bkwd_target_2 = nn.Linear(self.target_mem_slots_2, self.mem_context_size, bias=False)

        if USE_CUDA:
            self.M_k_fwd_target_2 = self.M_k_fwd_target_2.cuda()
            self.M_v_fwd_target_2 = self.M_v_fwd_target_2.cuda()
            self.M_k_bkwd_target_2 = self.M_k_bkwd_target_2.cuda()
            self.M_v_bkwd_target_2 = self.M_v_bkwd_target_2.cuda()
        
    def add_target_pad_3(self):
        self.M_k_fwd_target_3 = nn.Linear(self.hidden_size, self.target_mem_slots_3, bias=False)
        self.M_v_fwd_target_3 = nn.Linear(self.target_mem_slots_3, self.mem_context_size, bias=False)
        self.M_k_bkwd_target_3 = nn.Linear(self.hidden_size, self.target_mem_slots_3, bias=False)
        self.M_v_bkwd_target_3 = nn.Linear(self.target_mem_slots_3, self.mem_context_size, bias=False)

        if USE_CUDA:
            self.M_k_fwd_target_3 = self.M_k_fwd_target_3.cuda()
            self.M_v_fwd_target_3 = self.M_v_fwd_target_3.cuda()
            self.M_k_bkwd_target_3 = self.M_k_bkwd_target_3.cuda()
            self.M_v_bkwd_target_3 = self.M_v_bkwd_target_3.cuda()

    def add_target_pad_4(self):
        self.M_k_fwd_target_4 = nn.Linear(self.hidden_size, self.target_mem_slots_4, bias=False)
        self.M_v_fwd_target_4 = nn.Linear(self.target_mem_slots_4, self.mem_context_size, bias=False)
        self.M_k_bkwd_target_4 = nn.Linear(self.hidden_size, self.target_mem_slots_4, bias=False)
        self.M_v_bkwd_target_4 = nn.Linear(self.target_mem_slots_4, self.mem_context_size, bias=False)

        if USE_CUDA:
            self.M_k_fwd_target_4 = self.M_k_fwd_target_4.cuda()
            self.M_v_fwd_target_4 = self.M_v_fwd_target_4.cuda()
            self.M_k_bkwd_target_4 = self.M_k_bkwd_target_4.cuda()
            self.M_v_bkwd_target_4 = self.M_v_bkwd_target_4.cuda()


    def forward(self, input_seqs, input_lengths, hidden_f=None, hidden_b=None):
        max_input_length, batch_size = input_seqs.size()

        lengths2 = Variable(torch.LongTensor(input_lengths))
        mask = sequence_mask(lengths2, max_len=max_input_length).transpose(0, 1).unsqueeze(2)

        forward_outs = Variable(torch.zeros(max_input_length, batch_size, self.hidden_size))
        backward_outs = Variable(torch.zeros(max_input_length, batch_size, self.hidden_size))
        forward_hiddens = Variable(torch.zeros(max_input_length, batch_size, self.hidden_size))
        dummy_tensor = Variable(torch.zeros(1, batch_size, self.mem_context_size))

        if USE_CUDA:
            lengths2 = lengths2.cuda()
            mask = mask.cuda()
            forward_outs = forward_outs.cuda()
            backward_outs = backward_outs.cuda()
            forward_hiddens = forward_hiddens.cuda()
            dummy_tensor = dummy_tensor.cuda()

        #forward pass
        for i in range(max_input_length):
            seq_i = input_seqs[i]
            embedded = self.embedding(seq_i)
            embedded = self.dropout(embedded)
            embedded = embedded.unsqueeze(0)  # S=1 x B x N


            # Calculate attention from memory:
            ######################################
            if hidden_f is not None:
                if self.M_k_fwd_target is None:
                    alpha_tilde = torch.exp(self.M_k_fwd(hidden_f.squeeze(0))) # batch x num_slots
                elif self.M_k_fwd_target is not None and self.M_k_fwd_target_2 is None:
                    alpha_source = torch.exp(self.M_k_fwd(hidden_f.squeeze(0)))
                    alpha_target = torch.exp(self.M_k_fwd_target(hidden_f.squeeze(0)))
                    alpha_tilde = torch.cat((alpha_source, alpha_target), 1) # batch x (old+new slots)
                elif self.M_k_fwd_target is not None and self.M_k_fwd_target_2 is not None and self.M_k_fwd_target_3 is None:
                    alpha_source = torch.exp(self.M_k_fwd(hidden_f.squeeze(0)))
                    alpha_target = torch.exp(self.M_k_fwd_target(hidden_f.squeeze(0)))
                    alpha_target_2 = torch.exp(self.M_k_fwd_target_2(hidden_f.squeeze(0)))
                    alpha_tilde = torch.cat((alpha_source, alpha_target, alpha_target_2), 1)
                elif self.M_k_fwd_target is not None and self.M_k_fwd_target_2 is not None and self.M_k_fwd_target_3 is not None and self.M_k_fwd_target_4 is None:
                    alpha_source = torch.exp(self.M_k_fwd(hidden_f.squeeze(0)))
                    alpha_target = torch.exp(self.M_k_fwd_target(hidden_f.squeeze(0)))
                    alpha_target_2 = torch.exp(self.M_k_fwd_target_2(hidden_f.squeeze(0)))
                    alpha_target_3 = torch.exp(self.M_k_fwd_target_3(hidden_f.squeeze(0)))
                    alpha_tilde = torch.cat((alpha_source, alpha_target, alpha_target_2, alpha_target_3), 1) 
                else:
                    alpha_source = torch.exp(self.M_k_fwd(hidden_f.squeeze(0)))
                    alpha_target = torch.exp(self.M_k_fwd_target(hidden_f.squeeze(0)))
                    alpha_target_2 = torch.exp(self.M_k_fwd_target_2(hidden_f.squeeze(0)))
                    alpha_target_3 = torch.exp(self.M_k_fwd_target_3(hidden_f.squeeze(0)))
                    alpha_target_4 = torch.exp(self.M_k_fwd_target_4(hidden_f.squeeze(0)))
                    alpha_tilde = torch.cat((alpha_source, alpha_target, alpha_target_2, alpha_target_3, alpha_target_4), 1)

                alpha_sum = torch.sum(alpha_tilde, dim = 1).view(-1, 1) # batch x 1
                alpha = torch.div(alpha_tilde, alpha_sum.expand(alpha_tilde.size())) # batch x num_slots

                if self.M_v_fwd_target is None: 
                    memory_context = self.M_v_fwd(alpha) # batch x context
                elif self.M_v_fwd_target is not None and self.M_v_fwd_target_2 is None:
                    mem_context_source = self.M_v_fwd(alpha[:, :self.num_mem_slots]) # batch x old_slots
                    mem_context_target = self.M_v_fwd_target(alpha[:, self.num_mem_slots:])
                    memory_context = mem_context_source + mem_context_target
                elif self.M_v_fwd_target is not None and self.M_v_fwd_target_2 is not None and self.M_v_fwd_target_3 is None:
                    mem_context_source = self.M_v_fwd(alpha[:, :self.num_mem_slots]) # batch x old_slots
                    mem_context_target = self.M_v_fwd_target(alpha[:, self.num_mem_slots:self.num_mem_slots+self.target_mem_slots])
                    mem_context_target_2 = self.M_v_fwd_target_2(alpha[:, self.num_mem_slots+self.target_mem_slots:self.num_mem_slots+self.target_mem_slots+self.target_mem_slots_2])
                    memory_context = mem_context_source + mem_context_target + mem_context_target_2
                elif self.M_v_fwd_target is not None and self.M_v_fwd_target_2 is not None and self.M_v_fwd_target_3 is not None and self.M_v_fwd_target_4 is None:
                    mem_context_source = self.M_v_fwd(alpha[:, :self.num_mem_slots])
                    mem_context_target = self.M_v_fwd_target(alpha[:, self.num_mem_slots:self.num_mem_slots+self.target_mem_slots])
                    mem_context_target_2 = self.M_v_fwd_target_2(alpha[:, self.num_mem_slots+self.target_mem_slots:self.num_mem_slots+self.target_mem_slots+self.target_mem_slots_2])
                    mem_context_target_3 = self.M_v_fwd_target_3(alpha[:, self.num_mem_slots+self.target_mem_slots+self.target_mem_slots_2:self.num_mem_slots+self.target_mem_slots+self.target_mem_slots_2+self.target_mem_slots_3])
                    memory_context = mem_context_source + mem_context_target + mem_context_target_2 + mem_context_target_3
                else:
                    mem_context_source = self.M_v_fwd(alpha[:, :self.num_mem_slots]) 
                    mem_context_target = self.M_v_fwd_target(alpha[:, self.num_mem_slots:self.num_mem_slots+self.target_mem_slots])
                    mem_context_target_2 = self.M_v_fwd_target_2(alpha[:, self.num_mem_slots+self.target_mem_slots:self.num_mem_slots+self.target_mem_slots+self.target_mem_slots_2])
                    mem_context_target_3 = self.M_v_fwd_target_3(alpha[:, self.num_mem_slots+self.target_mem_slots+self.target_mem_slots_2:self.num_mem_slots+self.target_mem_slots+self.target_mem_slots_2+self.target_mem_slots_3])
                    mem_context_target_4 = self.M_v_fwd_target_4(alpha[:, self.num_mem_slots+self.target_mem_slots+self.target_mem_slots_2+self.target_mem_slots_3:self.num_mem_slots+self.target_mem_slots+self.target_mem_slots_2+self.target_mem_slots_3+self.target_mem_slots_4])
                    memory_context = mem_context_source + mem_context_target + mem_context_target_2 + mem_context_target_3 + mem_context_target_4

                if USE_CUDA:
                    memory_context = memory_context.cuda()
                ######################################

                rnn_input = torch.cat( (embedded, memory_context.unsqueeze(0)), 2 )
                output_f, hidden_f_new = self.forward_rnn(rnn_input, hidden_f)
            else:
                rnn_input = torch.cat((embedded, dummy_tensor), 2)
                output_f, hidden_f_new = self.forward_rnn(rnn_input, hidden_f)

            output_f = torch.tanh(output_f)
            hidden_f = hidden_f_new            
            forward_hiddens[i,:,:] = hidden_f[-1,:,:]
            forward_outs[i] = output_f.squeeze(0)

        #backward pass
        for i in range(1,max_input_length+1):
            seq_i = input_seqs[-i]
            embedded = self.embedding(seq_i)
            embedded = self.dropout(embedded)
            embedded = embedded.unsqueeze(0)  # S=1 x B x N


            # Calculate attention from memory:
            ######################################
            if hidden_b is not None:
                if self.M_k_bkwd_target is None:
                    alpha_tilde = torch.exp(self.M_k_bkwd(hidden_b.squeeze(0))) # batch x num_slots
                elif self.M_k_bkwd_target is not None and self.M_k_bkwd_target_2 is None:
                    alpha_source = torch.exp(self.M_k_bkwd(hidden_b.squeeze(0)))
                    alpha_target = torch.exp(self.M_k_bkwd_target(hidden_b.squeeze(0)))
                    alpha_tilde = torch.cat((alpha_source, alpha_target), 1)
                elif self.M_k_bkwd_target is not None and self.M_k_bkwd_target_2 is not None and self.M_k_bkwd_target_3 is None:
                    alpha_source = torch.exp(self.M_k_bkwd(hidden_f.squeeze(0)))
                    alpha_target = torch.exp(self.M_k_bkwd_target(hidden_f.squeeze(0)))
                    alpha_target_2 = torch.exp(self.M_k_bkwd_target_2(hidden_f.squeeze(0)))
                    alpha_tilde = torch.cat((alpha_source, alpha_target, alpha_target_2), 1)
                elif self.M_k_bkwd_target is not None and self.M_k_bkwd_target_2 is not None and self.M_k_bkwd_target_3 is not None and self.M_k_bkwd_target_4 is None:
                    alpha_source = torch.exp(self.M_k_bkwd(hidden_f.squeeze(0)))
                    alpha_target = torch.exp(self.M_k_bkwd_target(hidden_f.squeeze(0)))
                    alpha_target_2 = torch.exp(self.M_k_bkwd_target_2(hidden_f.squeeze(0)))
                    alpha_target_3 = torch.exp(self.M_k_bkwd_target_3(hidden_f.squeeze(0)))
                    alpha_tilde = torch.cat((alpha_source, alpha_target, alpha_target_2, alpha_target_3), 1)
                else:
                    alpha_source = torch.exp(self.M_k_bkwd(hidden_f.squeeze(0)))
                    alpha_target = torch.exp(self.M_k_bkwd_target(hidden_f.squeeze(0)))
                    alpha_target_2 = torch.exp(self.M_k_bkwd_target_2(hidden_f.squeeze(0)))
                    alpha_target_3 = torch.exp(self.M_k_bkwd_target_3(hidden_f.squeeze(0)))
                    alpha_target_4 = torch.exp(self.M_k_bkwd_target_4(hidden_f.squeeze(0)))
                    alpha_tilde = torch.cat((alpha_source, alpha_target, alpha_target_2, alpha_target_3, alpha_target_4), 1)

                alpha_sum = torch.sum(alpha_tilde, dim = 1).view(-1, 1) # batch x 1
                alpha = torch.div(alpha_tilde, alpha_sum.expand(alpha_tilde.size())) # batch x num_slots
                if self.M_v_bkwd_target is None: 
                    memory_context = self.M_v_bkwd(alpha) # batch x context
                elif self.M_v_bkwd_target is not None and self.M_v_bkwd_target_2 is None:
                    mem_context_source = self.M_v_bkwd(alpha[:, :self.num_mem_slots])
                    mem_context_target = self.M_v_bkwd_target(alpha[:, self.num_mem_slots:])
                    memory_context = mem_context_source + mem_context_target

                elif self.M_v_bkwd_target is not None and self.M_v_bkwd_target_2 is not None and self.M_v_bkwd_target_3 is None:
                    mem_context_source = self.M_v_bkwd(alpha[:, :self.num_mem_slots])
                    mem_context_target = self.M_v_bkwd_target(alpha[:, self.num_mem_slots:self.num_mem_slots+self.target_mem_slots])
                    mem_context_target_2 = self.M_v_bkwd_target_2(alpha[:, self.num_mem_slots+self.target_mem_slots:self.num_mem_slots+self.target_mem_slots+self.target_mem_slots_2])
                    memory_context = mem_context_source + mem_context_target + mem_context_target_2
                elif self.M_v_bkwd_target is not None and self.M_v_bkwd_target_2 is not None and self.M_v_bkwd_target_3 is not None and self.M_v_bkwd_target_4 is None:
                    mem_context_source = self.M_v_bkwd(alpha[:, :self.num_mem_slots])
                    mem_context_target = self.M_v_bkwd_target(alpha[:, self.num_mem_slots:self.num_mem_slots+self.target_mem_slots])
                    mem_context_target_2 = self.M_v_bkwd_target_2(alpha[:, self.num_mem_slots+self.target_mem_slots:self.num_mem_slots+self.target_mem_slots+self.target_mem_slots_2])
                    mem_context_target_3 = self.M_v_bkwd_target_3(alpha[:, self.num_mem_slots+self.target_mem_slots+self.target_mem_slots_2:self.num_mem_slots+self.target_mem_slots+self.target_mem_slots_2+self.target_mem_slots_3])
                    memory_context = mem_context_source + mem_context_target + mem_context_target_2 + mem_context_target_3
                else:
                    mem_context_source = self.M_v_bkwd(alpha[:, :self.num_mem_slots])
                    mem_context_target = self.M_v_bkwd_target(alpha[:, self.num_mem_slots:self.num_mem_slots+self.target_mem_slots])
                    mem_context_target_2 = self.M_v_bkwd_target_2(alpha[:, self.num_mem_slots+self.target_mem_slots:self.num_mem_slots+self.target_mem_slots+self.target_mem_slots_2])
                    mem_context_target_3 = self.M_v_bkwd_target_3(alpha[:, self.num_mem_slots+self.target_mem_slots+self.target_mem_slots_2:self.num_mem_slots+self.target_mem_slots+self.target_mem_slots_2+self.target_mem_slots_3])
                    mem_context_target_4 = self.M_v_bkwd_target_4(alpha[:, self.num_mem_slots+self.target_mem_slots+self.target_mem_slots_2+self.target_mem_slots_3:self.num_mem_slots+self.target_mem_slots+self.target_mem_slots_2+self.target_mem_slots_3+self.target_mem_slots_4])
                    memory_context = mem_context_source + mem_context_target + mem_context_target_2 + mem_context_target_3 + mem_context_target_4


                if USE_CUDA:
                    memory_context = memory_context.cuda()
                ######################################

                rnn_input = torch.cat( (embedded, memory_context.unsqueeze(0)), 2 )
                output_b, hidden_b_new = self.backward_rnn(rnn_input, hidden_b)
            else:
                rnn_input = torch.cat((embedded, dummy_tensor), 2)
                output_b, hidden_b_new = self.backward_rnn(rnn_input, hidden_b)

            output_b = torch.tanh(output_b)
            hidden_b = hidden_b_new
            back_mask = mask[-i, :].unsqueeze(0).expand_as(hidden_b)
            hidden_b = hidden_b * back_mask.float()
            backward_outs[-i] = output_b.squeeze(0)

        masked_forward_outs = forward_outs * mask.float()
        masked_backward_outs = backward_outs * mask.float()
        masked_output = masked_forward_outs + masked_backward_outs #S x B x H
        
        hiddens_out = Variable(torch.zeros(2, batch_size, self.hidden_size))
        if USE_CUDA:
            hiddens_out = hiddens_out.cuda()
        for i in range(batch_size):
            hiddens_out[0,i,:] = forward_hiddens[input_lengths[i]-1, i, :]
        
        hiddens_out[1,:,:] = hidden_b[-1,:,:]
        
        return masked_forward_outs, masked_backward_outs, hiddens_out


    def init_hidden(self, batch_size):
        if self.base_rnn == nn.LSTM:
            return self._init_LSTM(batch_size)
        elif self.base_rnn == nn.GRU:
            return self._init_GRU(batch_size)
        else:
            raise NotImplementedError()

    def _init_GRU(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if USE_CUDA:
            return result.cuda()
        else:
            return result

    def _init_LSTM(self, batch_size):
        result = (Variable(torch.zeros(1, batch_size, self.hidden_size)), Variable(torch.zeros(1, batch_size, self.hidden_size)))
        if USE_CUDA:
            return result.cuda()
        else:
            return result


    def extend_embedding_layer(self, w2i_dict, num_new_words):
        glove_vecs = load_glove_embeddings(w2i_dict, self.hidden_size) 
        
        new_embedding_layer = nn.Embedding(self.input_size+num_new_words, self.hidden_size)
        new_embedding_layer.weight = nn.Parameter(glove_vecs, requires_grad=True)
        added_rows = new_embedding_layer.weight[self.input_size:, :]
        if USE_CUDA:
            new_embedding_layer = new_embedding_layer.cuda()
            added_rows = added_rows.cuda()
        final_weights = torch.cat((self.embedding.weight, added_rows), 0)
        self.embedding = nn.Embedding(self.input_size+num_new_words, self.hidden_size)
        self.embedding.weight = nn.Parameter(final_weights.data, requires_grad=True)
        self.input_size = self.input_size + num_new_words
        if USE_CUDA:
            self.embedding = self.embedding.cuda()

