import torch
from torch import nn
from transformers import BertModel
from to_torch_dataset import Dataset
from torch.optim import Adam
from tqdm import tqdm


class BertClassifier(nn.Module):
    '''
    a Bert model which classify word to topics using a 12-layer model.
    param: num_output(int): number of output labels
    param: dropout(float): optional
    '''
    def __init__(self, num_output, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_output)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def train(model, train_data, val_data, learning_rate=1e-6, epochs=5, batch_size=16):
    '''
    the train function used to train the BERT model.
    param: model: the BERT model 
    param: train_data(dataframe): the dataframe used to train the data. Must contain text and label as columns
    param: val_data(dataframe): the dataframe used to validate the data. Must contain text and label as columns
    param: learning_rate(float): default=1e6
    param: epochs(int): default=5
    param: batch_size(int): default=16

    return: acc_train_lst(list): a list of training accuracy
    return: acc_val_lst(list): a list of validation accuracy
    return: loss_train_lst(list): a list of training loss
    return: loss_val_lst(list): a list of validation loss
    '''
    acc_train_lst, acc_val_lst, loss_train_lst, loss_val_lst = [], [], [], []

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.type(torch.LongTensor)
                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.type(torch.LongTensor)
                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')

            acc_train_lst.append(total_acc_train / len(train_data))
            acc_val_lst.append(total_acc_val / len(val_data))
            loss_train_lst.append(total_loss_train / len(train_data))
            loss_val_lst.append(total_loss_val / len(val_data))

    return model, acc_train_lst, acc_val_lst, loss_train_lst, loss_val_lst
            
                  

def evaluate(model, test_data, batch_size=16):
    '''
    an evaluate function to test the trained BERT model
    param: trained BERT model
    param: test_data(dataframe): a test data containing text and label as columns
    
    return: model
    return: test accuracy
    '''
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)

              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc
    
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')

    return model, total_acc_test / len(test_data)
    
