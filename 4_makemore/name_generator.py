import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
import random

# Critical Properties
file_path       =   "names.txt"     # File path to the training data
block_size      =   5               # This is the context length of the model
data_set_distrubution = (0.8, 0.9)  # Training: 0-n1 Validation: n1-n2 Test: n2-∞
 
seed = [42, 2147483647]             # Random seed for reproducibility, s1: data distrubution, s2: NN weights & biases
mlp_struct      =   [27,            # Vocabulary
                    10,             # Vocab distrubution dimension
                    200,            # Hidden layer neuron count
                    27]             # Classification
train_count     =   300000           # Amount of times trained for
batch_size      =   256             # Amount of examples per batch
steps_o_steps   =   6               # Amount of times training speed gets halved
lr              =   0.1             # Initial learning rate
lr_rate         =   0.5             # Each depth the lr gets multiplied by this number
config_path     =   "config.txt"    # Path of the config file (for graph name storage)
runtime = -1                        # Runtime in seconds
plt_count = -1                      # unique id of this model


# Properties
training_update_frequence = 0.01    # 0.01 = 1%
name_generation_count = 40          # Amount of names to generate

# Global variables
words       =   None                # All the names in a list
stoi, itos  =   None, None          # For encoding/decoding
Xtr, Ytr    =   None, None          # Training
Xdev, Ydev  =   None, None          # Develeopment (eval)
Xte, Yte    =   None, None          # Test (final eval)
parameters  =   None                # All the weights, biases, layers of the NN
    # p1:   =   None                # C lookup table
    # p2:   =   None                # W1 weights of the hidden layer
    # p3:   =   None                # b1 biases of the hidden layer
    # p4:   =   None                # W2 weights of the output layer
    # p5:   =   None                # b2 biases of the output layer
parameter_count = -1                # Count of all the parameters in the NN (for benchmark)
lri, lossi      = [], []            # PLACEHOLDER
stepi, loss_eva = [], []            # PLACEHOLDER
last_real_loss  = -1                # The last 'real' loss value

def build_decoding_table(): # Builds the decoding table, "a <-> 1"
    global words, stoi, itos
    words = open(file_path, 'r').read().splitlines()
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}


def build_dataset(words_t): # Builds the entire dataset
    global block_size
    X, Y = [], []
    for w in words_t:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

def build_subsets(): # Builds subsets with random distrubution (training, eva...)
    global seed, words, data_set_distrubution, Xtr, Ytr, Xdev, Ydev, Xte, Yte
    random.seed(seed[0])
    random.shuffle(words)
    n1 = int(data_set_distrubution[0]*len(words))
    n2 = int(data_set_distrubution[1]*len(words))
    Xtr, Ytr = build_dataset(words[:n1])
    Xdev, Ydev = build_dataset(words[n1:n2])
    Xte, Yte = build_dataset(words[n2:])

def set_training_data(): # Builds all the data for training and evaluation.
    build_decoding_table()
    build_subsets()

def create_mlp(): # Creates the MLP in pytorch
    global parameters, parameter_count
    g = torch.Generator().manual_seed(seed[1])
    C = torch.randn((mlp_struct[0], mlp_struct[1]), generator=g)
    W1 = torch.randn((block_size*mlp_struct[1], 200), generator=g)
    b1 = torch.randn(mlp_struct[2], generator=g)
    W2 = torch.randn((mlp_struct[2], mlp_struct[3]), generator=g)
    b2 = torch.randn(mlp_struct[3], generator=g)
    parameters = [C, W1, b1, W2, b2]
    parameter_count = sum(p.nelement() for p in parameters)

    # Very important
    for p in parameters:
        p.requires_grad = True  

def train():
    global parameters, last_real_loss, runtime, lr, stepi, lossi, loss_eva
    start = time.time()
    for i in range(train_count):
        # Minibatch construct
        ix = torch.randint(0, Xtr.shape[0], (batch_size,))
        
        # Create pointers for readibility
        C = parameters[0]
        W1 = parameters[1]
        b1 = parameters[2]
        W2 = parameters[3]
        b2 = parameters[4]

        # Forward pass
        emb = C[Xtr[ix]]
        h = torch.tanh(emb.view(-1, block_size*mlp_struct[1]) @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Ytr[ix])
        
        # Backward pass
        for p in parameters:
            p.grad = None
        loss.backward()
        
        # Update
        if (i%(10.0/steps_o_steps) == 0):
            lr = lr * lr_rate
        
        # Apply Gradients
        for p in parameters:
            p.data += -lr * p.grad

        # Track stats
        stepi.append(i)
        lossi.append(loss.log10().item())

        # Training progress bar
        if i%(train_count*training_update_frequence) == 0:
            percentage = i/(train_count*0.1)
            print("<", '='*int(percentage), ' '*int(10-percentage) ,">", round((percentage*10), 1), "%")

        # Track loss on eva data
        if i%(train_count*training_update_frequence) == 0:
            emb = C[Xdev]
            h = torch.tanh(emb.view(-1, block_size*mlp_struct[1]) @ W1 + b1)
            logits = h @ W2 + b2
            loss = F.cross_entropy(logits, Ydev)
            loss_eva.append(loss.log10().item())
            last_real_loss = loss.item()

    runtime = time.time() - start
    print("finished")

def save_graph():
    global plt_count
    # Read last saved serial number
    f = open(config_path, 'r')
    plt_count = int(f.read())
    f.close()

    # Plot and save graph
    plt.plot(stepi, lossi)
    plt.plot(np.arange(0, train_count, train_count/100), loss_eva)
    plt.title((f'a-{round(last_real_loss, 3)} c-{train_count} bs-{batch_size} rt-{round(runtime, 0)}s cl-{block_size} sos-{steps_o_steps}'))
    plt.savefig(f'models/{plt_count}_graph.png')

    # Increment plt_count in memory and config
    plt_count = plt_count +1
    f = open(config_path, 'w')
    f.write(str(plt_count))
    f.close()

def generate_names():
    # Create pointers for readibility
    C = parameters[0]
    W1 = parameters[1]
    b1 = parameters[2]
    W2 = parameters[3]
    b2 = parameters[4]

    # Sample from the model
    g = torch.Generator().manual_seed(seed[1])
    f = open(f'models/{plt_count-1}_properties.txt', 'a')
    f.write(f'----------------------------\n')
    f.write(f'Generated Names\n')
    name = ""

    for _ in range(name_generation_count):
        out = []
        context = [0] * block_size # initialize with all ...
        while True:
            emb = C[torch.tensor([context])] # (1,block_size,d)
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        
        #print()
        name += ''.join(itos[i] for i in out) + '\n'
    f.write(f'{name}\n')
    f.close()

def save_model_properties():
    f = open(f'models/{plt_count-1}_properties.txt', 'a')

    f.write(f'file_path = {file_path}\n')
    f.write(f'block_size = {block_size}\n')
    f.write(f'data_set_distribution = {data_set_distrubution}\n')
    f.write(f'seed = {seed}\n')
    f.write(f'mlp_struct = {mlp_struct}\n')
    f.write(f'train_count = {train_count}\n')
    f.write(f'batch_size = {batch_size}\n')
    f.write(f'steps_o_steps = {steps_o_steps}\n')
    f.write(f'lr = {lr}\n')
    f.write(f'lr_rate = {lr_rate}\n')
    f.write(f'config_path = {config_path}\n')
    f.write(f'runtime = {runtime}s\n')
    f.write(f'plt_count = {plt_count}\n')

    f.close()

def main(): 
    set_training_data()
    create_mlp()
    train()
    save_graph()
    save_model_properties()
    generate_names()

if __name__=="__main__": 
    main() 
