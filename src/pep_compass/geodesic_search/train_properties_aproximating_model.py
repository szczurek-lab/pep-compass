import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


def get_experiment_name(args):
    if args.experiment_name != None:
        return "PW_" + args.experiment_name + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        return "PW_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format daty

def get_formatted_date():
    """
    Returns the current date in the format: month-day-hour (MM-DD-HH).
    """
    return datetime.now().strftime("%m-%d-%H-%M")

def print_model_info(model: torch.nn.Module) -> None:
    """
    Prints the number of parameters and the size of a PyTorch model in MB.
    
    Args:
        model (torch.nn.Module): The PyTorch model.
    """
    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 * 1024)
    
    print(f"Model parameters: {num_params:,}")
    print(f"Model size: {model_size_mb:.2f} MB")

def initialize_wandb(args, experiment_name):
    # Inicjalizacja projektu
    wandb.init(project=args.project_name, name=experiment_name)

    # Logowanie hiperparametrów
    wandb.config.update(args)

def finish_wandb():
    wandb.finish()

class Distribution_Based_Regression_Transformer(nn.Module):
    '''
    A model which takes in distributions over peptides (so a vector of shape (525)) and returns a vector of predicted means and predicted
    variances, so two vectors of shape (8). It does so by prepending the start token to the data and then lineary projecting the start token onto desired 
    vectors at the end

    model_dim = dimension of transformer embedings. If model dim != 21, then inputs are lineary projected to model dim at the begining
    
    Input vectors are reshaped to (BS, 25, 21)
    Processed vectors are of shape (BS, 25, MODEL DIM)
    '''
    def __init__(self, model_dim, num_heads, num_layers, hidden_ff_dim, dropout=0.1, number_of_predictions = 16, seq_len = 25,
                  alphabet_size = 21, device= 'cpu'):
        super(Distribution_Based_Regression_Transformer, self).__init__()
        self.model_dim = model_dim
        self.seq_len = seq_len #sequence length not counting the start token
        self.alphabet_size = alphabet_size
        self.loss = torch.nn.MSELoss()
        self.device = device
        self.add_additional_calculation_tokens = False # set to false by default
        
        # Initial linear to change size to model_dim:
        self.initial_linear = nn.Linear(alphabet_size, model_dim, device = self.device)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len + 1, model_dim, device=self.device))  # sequence length 25 + 1 start token
        self.start_token = nn.Parameter(torch.randn(1, 1, model_dim, device=self.device))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=hidden_ff_dim,
            dropout=dropout,
            batch_first=True,
            device=self.device
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor = False)
        
        # Regression head
        self.fc = nn.Linear(model_dim, number_of_predictions, device=self.device)

    def initial_reshape(self, x):
        reshaped_tensor = x.reshape(x.shape[0], self.seq_len, self.alphabet_size)
        return reshaped_tensor
        

    def forward(self, x): #Zrobić większy model, model 21 wymiarowy jest zbyt mały
        #Initial x.shape = [BS, 525]
        x = self.initial_reshape(x)

        #x.shape = [BS, SEQ_LEN, ALPHABET SIZE]
        if self.model_dim != 21: #hard coded because this is the number of aminoacids + padding character
            x = self.initial_linear(x)
        
        # x.shape = [BS, SEQ_LEN, MODEL_DIM]

        # now lets add the start token and the calc_tokens
        if self.add_additional_calculation_tokens == True:
            x = torch.cat([self.calc2_token.expand((x.shape[0], 1, self.model_dim)), x], dim=1)
            x = torch.cat([self.calc1_token.expand((x.shape[0], 1, self.model_dim)), x], dim=1)
        x = torch.cat([self.start_token.expand((x.shape[0], 1, self.model_dim)), x], dim=1)
        x = x + self.positional_encoding

        # x.shape = [BS, SEQL_LEN + 1, MODEL_DIM]

        # Pass through Transformer encoder
        x = self.encoder(x)

        x = x[:, 0, :] 
        output = self.fc(x)
        
        return output
    def prepare_dataloader(self, data, args, shuffle = True):
        return prepare_Distributional_dataloader(data, args, shuffle)
    
    def train_one_epoch(self, epoch, train_dataloader, val_dataloader, optimizer, device, do_validation, log_freq, val_freq, loss_fn):
        print('Starting epoch ', epoch)
        for i, new_data in enumerate(train_dataloader):
            inputs, targets = new_data
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = self(inputs)

            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            log_step = epoch * len(train_dataloader) + i

            if log_step % log_freq == 0:
                print(f'batch {i} loss {loss}')
                wandb.log({"train loss": loss}, step = log_step)

            if log_step % val_freq == 0 and do_validation:
                val_loss = self.validate_model(val_dataloader, device, loss_fn, args)
                print(f'batch {i} Validation loss : {val_loss}')
                wandb.log({"Validation loss": val_loss}, step = log_step)

    def validate_model(self, val_dataloader, device, loss_fn, args):
        self.eval()
        mean_val_loss = 0
        with torch.no_grad():
            for i, new_data in enumerate(val_dataloader):
                inputs, targets = new_data
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = self(inputs)

                loss = loss_fn(outputs, targets)
                mean_val_loss += loss.item() / len(val_dataloader)

            return mean_val_loss
        
    def evaluate_and_save(self, dataloader, device, output_path, save_results=False):
        """
        Evaluates the model on a dataset, optionally saves predictions and targets to a Parquet file,
        and logs average R2 and MAPE metrics to Weights & Biases.
        
        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader with input data.
            device (torch.device): Device to run the evaluation on.
            output_path (str): Path to save the Parquet file.
            save_results (bool): Whether to save predictions and targets to a Parquet file.
        """
        self.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for inputs, target in dataloader:
                inputs, target = inputs.to(device), target.to(device)
                output = self(inputs)
                
                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
        
        predictions = np.vstack(predictions)
        targets = np.vstack(targets)
        
        if save_results:
            df = pd.DataFrame({f"target_{i}": targets[:, i] for i in range(targets.shape[1])})
            df = df.assign(**{f"prediction_{i}": predictions[:, i] for i in range(predictions.shape[1])})
            df.to_parquet(output_path, index=False)
        
        r2_scores = [r2_score(targets[:, i], predictions[:, i]) for i in range(targets.shape[1])]
        # mape_scores = [mean_absolute_percentage_error(targets[:, i], predictions[:, i]) for i in range(targets.shape[1])]
        
        for i, (r2) in enumerate((r2_scores)):
            print(f"R2 Score for output {i}: {r2:.4f}")
            # print(f"MAPE for output {i}: {mape:.4f}")
        
        avg_r2 = np.mean(r2_scores)
        # avg_mape = np.mean(mape_scores)
        
        print(f"Average R2 Score: {avg_r2:.4f}")
        # print(f"Average MAPE: {avg_mape:.4f}")
        
        wandb.log({"Average R2": avg_r2})
    
class Distribution_Based_Distance_Predictor(Distribution_Based_Regression_Transformer):
    def __init__(self, model_dim, num_heads, num_layers, hidden_ff_dim, dropout=0.1, number_of_predictions = 16, seq_len = 25,
                  alphabet_size = 21, additional_calculation_tokens = False):
        super(Distribution_Based_Distance_Predictor, self).__init__(
            model_dim, num_heads, num_layers, hidden_ff_dim
        , dropout, number_of_predictions, seq_len, alphabet_size, additional_calculation_tokens)

    def prepare_dataloader(self, data, args, shuffle = True):
        return prepare_Distance_dataloader(data, args, shuffle)
    
    def train_one_epoch(self, epoch, train_dataloader, val_dataloader, optimizer, device, do_validation, log_freq, val_freq, loss_fn):
        print('Starting epoch ', epoch)
        for i, new_data in enumerate(train_dataloader):
            input1, input2, targets = new_data
            input1, input2, targets = input1.to(device), input2.to(device), targets.to(device)

            optimizer.zero_grad()

            output1 = self(input1)
            output2 = self(input2)

            distance = torch.norm(output1 - output2, dim=1)

            loss = loss_fn(distance, targets)
            loss.backward()
            optimizer.step()

            log_step = epoch * len(train_dataloader) + i

            if log_step % log_freq == 0:
                print(f'batch {i} loss {loss}')
                wandb.log({"train loss": loss}, step = log_step)

            if log_step % val_freq == 0 and do_validation:
                val_loss = self.validate_model(val_dataloader, device, loss_fn, args)
                print(f'batch {i} Validation loss : {val_loss}')
                wandb.log({"Validation loss": val_loss}, step = log_step)

    def validate_model(self, val_dataloader, device, loss_fn, args):
        self.eval()
        mean_val_loss = 0
        with torch.no_grad():
            for i, new_data in enumerate(val_dataloader):
                input1, input2, targets = new_data
                input1, input2, targets = input1.to(device), input2.to(device), targets.to(device)

                output1 = self(input1)
                output2 = self(input2)

                distance = torch.norm(output1 - output2, dim=1)
                loss = loss_fn(distance, targets)

                mean_val_loss += loss.item() / len(val_dataloader)

            return mean_val_loss

    def calculate_distance(self, distribution1, distribution2, device):
        distribution1 = distribution1.to(device)
        distribution2 = distribution2.to(device)
        return torch.norm(self(distribution1) - self(distribution2), dim=1)

    def evaluate_and_save(self, dataloader, device, output_path, save_results=False):
        """
        Evaluates the model on a dataset, optionally saves predictions and targets to a Parquet file,
        and logs average R2 and MAPE metrics to Weights & Biases.
        
        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader with input data.
            device (torch.device): Device to run the evaluation on.
            output_path (str): Path to save the Parquet file.
            save_results (bool): Whether to save predictions and targets to a Parquet file.
        """
        self.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for new_data in dataloader:
                input1, input2, target = new_data
                input1, input2, target = input1.to(device), input2.to(device), target.to(device)

                output1 = self(input1)
                output2 = self(input2)

                distance = torch.norm(output1 - output2, dim=1)
                
                predictions.append(distance.cpu().numpy())
                targets.append(target.cpu().numpy())
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        if save_results == True:
            df = pd.DataFrame({f"target": targets})
            df = df.assign(**{f"prediction": predictions})
            df.to_parquet(output_path, index=False)
            print('Results saved!')
        
        r2_scores = r2_score(targets, predictions)
        mape_scores = mean_absolute_percentage_error(targets, predictions)
        
        print(f'R2 score: {r2_scores}, MAPE: {mape_scores}')
        
        wandb.log({"R2": r2_scores})
        wandb.log({"MAPE": mape_scores})

class Distributional_Dataset(Dataset):
    '''
    Dataset of distributions, distrybution is the distrybution over peptides, target is the vector of 16 properties, means and variances
    '''
    def __init__(self, data, do_mean = True, do_var = False, do_MICs = False):
        self.data = data
        self.do_mean = do_mean
        self.do_var = do_var
        self.do_MICs = do_MICs

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.do_MICs == False:
            distribution = self.data['Distribution'][idx]
            mean = self.data['normalized_mean'][idx]
            var = self.data['normalized_variance'][idx]
            if self.do_mean == True and self.do_var == True:
                # print('Data consist of both means and variances')
                target = np.concatenate((mean, var))
            if self.do_mean == True and self.do_var == False:
                # print('Data consist of means')
                target = mean
            if self.do_mean== False and self.do_var == True:
                # print('Data consist of variances')
                target = var
            if self.do_mean == False and self.do_var == False:
                raise ValueError("Both do_mean and do_var cant be False!")
            return torch.tensor(distribution, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
        else:
            distribution = self.data['Original_Distribution'][idx]
            target = self.data['Beam_search_MICs_standardized'][idx]
            return torch.tensor(distribution, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

    
def prepare_dataloader(data, args, shuffle = True):
    dataset = Distributional_Dataset(data, args.do_mean, args.do_var, args.do_MICs)
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = shuffle)
    return dataloader

def prepare_Distance_dataloader(data, args, shuffle = True):
    dataset = Distance_Dataset(data, args.do_mean, args.do_var)
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = shuffle)
    return dataloader

def prepare_model(args):
    if args.model_type == 'properties':
        model = Distribution_Based_Regression_Transformer(args.model_dim, args.num_heads, args.num_layers, args.hidden_ff_dim,
                                                        args.dropout, args.number_of_predictions, seq_len = 25,
                                                        alphabet_size = 21,
                                                          add_aditional_calculation_tokens=args.additional_calculation_tokens)
    elif args.model_type == 'distance':
        model = Distribution_Based_Distance_Predictor(args.model_dim, args.num_heads, args.num_layers, args.hidden_ff_dim,
                                                        args.dropout, args.number_of_predictions, seq_len = 25,
                                                        alphabet_size = 21,
                                                          additional_calculation_tokens = args.additional_calculation_tokens)
        if args.number_of_predictions < 64:
            print('Remeber to set a sensible number of predictions, current number of predictions: ', args.number_of_predictions)
    else:
        raise ValueError('Dopuszczalne rodzaje modelu to tylko properties i distance!')
    return model

def train_model(train_data, val_data, device, args, do_validation, run_name):
    epochs = args.epochs
    log_freq = args.log_freq
    val_freq = args.val_freq

    model = prepare_model(args)
    print_model_info(model)
    model.to(device)

    model.train()

    train_dataloader = model.prepare_dataloader(train_data, args, shuffle = True)

    if do_validation:
        val_dataloader = model.prepare_dataloader(val_data, args, shuffle = False)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train_one_epoch(epoch, train_dataloader, val_dataloader, optimizer, device, do_validation, log_freq, val_freq, loss_fn)
                
        
    # Perform final validation
    if do_validation:
        final_val_loss = model.validate_model(val_dataloader, device, loss_fn, args)
        print('Training ended, final val loss: ', final_val_loss)
        wandb.log({"Final Validation loss": final_val_loss})
        model.evaluate_and_save(val_dataloader, device,
                           "saved_results/"  + run_name + "_results_" + ".parquet",
                             args.save_results)

    # Save the model
    if args.save_model:
        torch.save(model.state_dict(), "saved_models/" + run_name + "_state_dict_" + ".pth")
        model_config = {
            "model_dim": args.model_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "hidden_ff_dim": args.hidden_ff_dim,
            "dropout": args.dropout,
            "number_of_predictions": args.number_of_predictions,
            "seq_len": 25,
            "alphabet_size": 21,
            "add_adnitional_calculation_tokens": args.additional_calculation_tokens
        }
        with open("saved_models/" + run_name + "_config.json", "w") as f:
            json.dump(model_config, f)  # Save the config as a json file
        print('Model succesfully saved!')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    elif v.lower() in ("false", "0", "no"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    parser = argparse.ArgumentParser(description="Program skeleton")
    parser.add_argument("--model_dim", type=int, help="Model embedding dim", required=False, default=256)
    parser.add_argument("--num_heads", type=int, help="Number of attention heads", required=False, default=16)
    parser.add_argument("--num_layers", type=int, help="Depth of the transformer model", required=False, default=16)
    parser.add_argument("--hidden_ff_dim", type=int, help="Dimension of the hidden transformer ff layer", required=False, default=512)
    parser.add_argument("--dropout", type=float, help="Dropout", required=False, default=0.05)
    parser.add_argument("--number_of_predictions", type=int, help="Number of predictions that the model is to produce", required=False, default=8)
    parser.add_argument("--epochs", type=int, help="epochs", required=False, default=20)
    parser.add_argument("--log_freq", type=int, help="log_freq", required=False, default=100)
    parser.add_argument("--val_freq", type=int, help="validation_freq", required=False, default=1000)
    parser.add_argument("--lr", type=float, help="Learning Rate", required=False, default=0.001)
    parser.add_argument("--train_data_path", type= str, help="Path to the training parquet file", required=True)
    parser.add_argument("--val_data_path", type= str, help="Path to the validation parquet file", required=False, default = None)
    parser.add_argument("--project_name", type= str, help="Wandb project_name", required=False, default = 'Magisterka')
    parser.add_argument("--experiment_name", type= str, help="Wandb experiment_name", required=False, default = None)
    parser.add_argument("--batch_size", type= int, help="batch_size", required=False, default = 256)
    parser.add_argument("--save_model", type= bool, help="Schould we save the model (deflaut false)", required=False, default = False)
    parser.add_argument("--save_results", type= str2bool, help="Schould we save resulting predictions of a final model? (deflaut false)", required=False, default = False)
    parser.add_argument("--do_mean", type= str2bool, help="Schould the model predict means:", required=False, default = True)
    parser.add_argument("--do_var", type= str2bool, help="Schould the model predict vars?", required=False, default = False)
    parser.add_argument("--do_MICs", type= str2bool, help="Schould the model predict MICs?", required=False, default = False)
    return parser.parse_args()

def main(args):
    run_name = get_experiment_name(args)
    initialize_wandb(args, run_name)
    train_data = pd.read_parquet(args.train_data_path)
    do_validation = False
    val_data = None
    if args.val_data_path != None:
        val_data = pd.read_parquet(args.val_data_path)
        do_validation = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Train data shape: ' ,train_data.shape, ' Using device: ', device)
    print('Wisible CUDA devices: ', torch.cuda.device_count())

    train_model(train_data, val_data, device, args, do_validation, run_name)

    finish_wandb()

if __name__ == "__main__":
    args = parse_args()
    main(args)
