import argparse
import math
import time
import torch
import torch.nn as nn
from net import gtnet
import numpy as np
import importlib
import random
from util import *
from trainer import Optim
import sys
from random import randrange
# import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import time

plt.rcParams['savefig.dpi'] = 1200

runs=10
#{x}= (e ^ [x'+log(x{-1} +1)]) -1


def inverse_diff_2d(output, I,shift):
    output[0,:]=torch.exp(output[0,:]+torch.log(I+shift))-shift
    for i in range(1,output.shape[0]):
        output[i,:]= torch.exp(output[i,:]+torch.log(output[i-1,:]+shift))-shift
    return output

def inverse_diff_3d(output, I,shift):
    output[:,0,:]=torch.exp(output[:,0,:]+torch.log(I+shift))-shift
    for i in range(1,output.shape[1]):
        output[:,i,:]=torch.exp(output[:,i,:]+torch.log(output[:,i-1,:]+shift))-shift
    return output

# def inverse_diff_2d(output, I):
#     output[0,:]+=I
#     for i in range(1,output.shape[0]):
#         output[i,:]+= output[i-1,:]
#     return output

# def inverse_diff_3d(output, I):
#     output[:,0,:]+=I
#     for i in range(1,output.shape[1]):
#         output[:,i,:]+= output[:,i-1,:]
#     return output

def plot_data(data,title):
    x=range(1,len(data)+1)
    plt.plot(x,data,'b-',label='Actual')
    plt.legend(loc="best",prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(title, y=1.03,fontsize=18)
    plt.ylabel("Trend",fontsize=15)
    plt.xlabel("Month",fontsize=15)
    locs, labs = plt.xticks() 
    plt.xticks(rotation='vertical',fontsize=13) 
    plt.yticks(fontsize=13)
    fig = plt.gcf()
    #plt.savefig('model/comparison/auto141/'+title[0:4]+str(counter+1)+'.png', bbox_inches="tight")

    plt.show()
    # plt.pause(4)
    # plt.close()


def consistent_name(name):

    if name=='CAPTCHA' or name=='DNSSEC' or name=='RRAM':
        return name

    #e.g., University of london
    if not name.isupper():
        words=name.split(' ')
        result=''
        for i,word in enumerate(words):
            if len(word)<=2: #e.g., "of"
                result+=word
            else:
                result+=word[0].upper()+word[1:]
            
            if i<len(words)-1:
                result+=' '

        return result
    

    words= name.split(' ')
    result=''
    for i,word in enumerate(words):
        if len(word)<=3 or '/' in word or word=='MITM' or word =='SIEM':
            result+=word
        else:
            result+=word[0]+(word[1:].lower())
        
        if i<len(words)-1:
            result+=' '
        
    return result

def save_metrics_1d(predict, test, title, type):
    #RRSE according to Lai et.al
    sum_squared_diff = torch.sum(torch.pow(test - predict, 2))
    root_sum_squared= math.sqrt(sum_squared_diff) #numerator

    #Relative Absolute Error RAE 
    sum_absolute_diff= torch.sum(torch.abs(test - predict))


##########################################################################################################

    
    test_s=test
    mean_all = torch.mean(test_s) # calculate the mean of each column in test call it Yj-
    diff_r = test_s - mean_all # subtract the mean from each element in the tensor test
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))# square the result and sum over all elements
    root_sum_squared_r=math.sqrt(sum_squared_r)#denominator

    #RRSE according to Lai et.al
    rrse=root_sum_squared/root_sum_squared_r
    #print('rrse=',root_sum_squared,'/',root_sum_squared_r)
###########################################################################################################
###########################################################################################################

    #Relative Absolute Error RAE 
    sum_absolute_r=torch.sum(torch.abs(diff_r))# absolute the result and sum over all elements
    rae=sum_absolute_diff/sum_absolute_r 
    rae=rae.item()
    title=title.replace('/','_')
    with open(type+'/'+title+'_'+type+'.txt',"w") as f:
      f.write('rse:'+str(rrse)+'\n')
      f.write('rae:'+str(rae)+'\n')
      f.close()



def plot_predicted_actual(predicted, actual, title, type,variance, confidence_95):

    #all months
    months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    M=[]
    for year in range (11,23):   
        for month in months:
            if year==11 and month not in ['Jul','Aug','Sep','Oct','Nov','Dec']:
                continue
            M.append(month+'-'+str(year))   
    M2=[]
    p=[]
    
    #last 3 years
    if type=='Testing':
        M=M[-len(predicted):]
        for index,value in enumerate(M):
            if 'Dec' in M[index] or 'Mar' in M[index] or 'Jun' in M[index] or 'Sep' in M[index]:
                M2.append(M[index])
                p.append(index+1) 
    
    else: ##Validation x axis: Oct-16 to Sep-19
        M=M[63:99]
        for index,value in enumerate(M):
            if 'Dec' in M[index] or 'Mar' in M[index] or 'Jun' in M[index] or 'Sep' in M[index]:
                M2.append(M[index])
                p.append(index+1) 

    x=range(1,len(predicted)+1)
    plt.plot(x,actual,'b-',label='Actual')
    plt.plot(x,predicted,'--', color='purple',label='Predicted')
    # Plot the confidence interval as a shaded region
    plt.fill_between(x, predicted-confidence_95.numpy(), predicted+confidence_95.numpy(), alpha=0.5, color='pink', label='95% Confidence')
    #plt.fill_between(x, predicted-variance.numpy(), predicted+variance.numpy(), alpha=0.1, color='purple', label='Variance')
    plt.legend(loc="best",prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    # if 'val' in type.lower():
    #     plt.title(title+' - '+type, y=1.03,fontsize=18)
    # else:
    plt.title(title, y=1.03,fontsize=18)
    plt.ylabel("Trend",fontsize=15)
    plt.xlabel("Month",fontsize=15)
    locs, labs = plt.xticks() 
    plt.xticks(ticks = p ,labels = M2, rotation='vertical',fontsize=13) 
    plt.yticks(fontsize=13)
    fig = plt.gcf()
    title=title.replace('/','_')
    plt.savefig(type+'/'+title+'_'+type+'.png', bbox_inches="tight")
    plt.savefig(type+'/'+title+'_'+type+".pdf", bbox_inches = "tight", format='pdf')


    plt.show(block=False)
    plt.pause(2)
    plt.close()



def s_mape(yTrue,yPred):
  mape=0
  for i in range(len(yTrue)):
    mape+= abs(yTrue[i]-yPred[i])/ (abs(yTrue[i])+abs(yPred[i]))
  mape/=len(yTrue)

  return mape



def evaluate_sliding_window(data, test_window, model, evaluateL2, evaluateL1, n_input, is_plot):
    #model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    variance=None
    confidence_95=None
    predictions = []
    sum_squared_diff=0
    sum_absolute_diff=0
    r=random.randint(0, 141)
    r=0
    print('testing r=',str(r))
    scale = data.scale.expand(test_window.size(0), data.m) #scale will have the max of each column (142 max values)
    # test_window_o=test_window*scale
    # test_window_o=inverse_diff_2d(test_window_o,data.inv_test_window[0])
    print('Test Window Feature:',test_window[:,r])
    
    

    x_input = test_window[0:n_input, :].clone() # Generate input sequence

    for i in range(n_input, test_window.shape[0],data.out_len): # X.shape[0]=n_input+36 (as example)?

        print('**************x_input*******************')
        # scale = data.scale.expand(x_input.size(0), data.m) #scale will have the max of each column (142 max values)
        # x_input_o=x_input*scale
        # x_input_o=inverse_diff_2d(x_input_o,data.inv_test_window[i-n_input,:])
        print(x_input[:,r])#prints 1 random column in the sliding window
        print('**************-------*******************')

        X = torch.unsqueeze(x_input,dim=0)
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        X = X.to(torch.float)


        y_true = test_window[i: i+data.out_len,:].clone() #10x142


        # Bayesian estimation
        num_runs = runs

        # Create a list to store the outputs
        outputs = []


        # Use model to predict next time step
        for _ in range(num_runs):
            with torch.no_grad():
                output = model(X)  
                y_pred = output[-1, :, :,-1].clone()#10x142
                #if this is the last predicted window and it exceeds the test window range
                if y_pred.shape[0]>y_true.shape[0]:
                    y_pred=y_pred[:-(y_pred.shape[0]-y_true.shape[0]),]
            outputs.append(y_pred)

        # Stack the outputs along a new dimension
        outputs = torch.stack(outputs)


        y_pred=torch.mean(outputs,dim=0)
        var = torch.var(outputs, dim=0)#variance
        std_dev = torch.std(outputs, dim=0)#standard deviation

        # Calculate 95% confidence interval
        z=1.96
        confidence=z*std_dev/torch.sqrt(torch.tensor(num_runs))



        #shift the sliding window
        #x_input = torch.cat([x_input[ data.out_len:, :], y_pred], dim=0)
        if data.P<=data.out_len:
            x_input = y_pred[-data.P:].clone()
        else:
            x_input = torch.cat([x_input[ -(data.P-data.out_len):, :].clone(), y_pred.clone()], dim=0)

        # scale = data.scale.expand(y_pred.size(0), data.m) #scale will have the max of each column (142 max values)

        # #inverse normalisation
        # y_pred*=scale
        # y_true*=scale

        #inverse difference
        # I=data.inv_test_window[i,:]
        # y_pred=inverse_diff_2d(y_pred,I)
        # y_true=inverse_diff_2d(y_true,I)

        print('----------------------------Predicted months',str(i-n_input+1),'to',str(i-n_input+data.out_len),'--------------------------------------------------')
        print(y_pred.shape,y_true.shape)
        y_pred_o=y_pred
        y_true_o=y_true
        for z in range(y_true.shape[0]):
            print(y_pred_o[z,r],y_true_o[z,r]) #only one col
        print('------------------------------------------------------------------------------------------------------------')


        if predict is None:
            predict = y_pred
            test = y_true
            variance=var
            confidence_95=confidence
        else:
            predict = torch.cat((predict, y_pred))
            test = torch.cat((test, y_true))
            variance=torch.cat((variance, var))
            confidence_95=torch.cat((confidence_95,confidence))

        # total_loss += evaluateL2(y_pred, y_true).item() #MSE
        # total_loss_l1 += evaluateL1(y_pred, y_true).item() #MAE
        # n_samples += (y_pred.size(0) * data.m)

    #     #RRSE according to Lai et.al
    #     sum_squared_diff += torch.sum(torch.pow(y_true - y_pred, 2))
    #     #Relative Absolute Error RAE 
    #     sum_absolute_diff+=torch.sum(torch.abs(y_true - y_pred))

    #rse = math.sqrt(total_loss / n_samples) / data.rse #Root Relative Squared Error (RRSE)?
    #rae = (total_loss_l1 / n_samples) / data.rae # Relative Absolute Error?

    scale = data.scale.expand(test.size(0), data.m) #scale will have the max of each column (142 max values)

    #inverse normalisation
    predict*=scale
    test*=scale
    variance*=scale
    confidence_95*=scale

    # #inverse difference
    # I=data.inv_test_window[n_input,:]
    # predict=inverse_diff_2d(predict,I,data.shift)
    # test=inverse_diff_2d(test,I,data.shift)


    #RRSE according to Lai et.al
    sum_squared_diff = torch.sum(torch.pow(test - predict, 2))
    #Relative Absolute Error RAE 
    sum_absolute_diff= torch.sum(torch.abs(test - predict))


##########################################################################################################
    #RRSE according to Lai et.al
    root_sum_squared= math.sqrt(sum_squared_diff) #numerator
    
    # scale = data.scale.expand(test.size(0), data.m) #scale will have the max of each column (142 max values)
    # test_s= test*scale
    test_s=test
    mean_all = torch.mean(test_s, dim=0) # calculate the mean of each column in test call it Yj-
    diff_r = test_s - mean_all.expand(test_s.size(0), data.m) # subtract the mean from each element in the tensor test
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))# square the result and sum over all elements
    root_sum_squared_r=math.sqrt(sum_squared_r)#denominator

    #RRSE according to Lai et.al
    rrse=root_sum_squared/root_sum_squared_r
    print('rrse=',root_sum_squared,'/',root_sum_squared_r)
###########################################################################################################
###########################################################################################################

    #Relative Absolute Error RAE 
    sum_absolute_r=torch.sum(torch.abs(diff_r))# absolute the result and sum over all elements
    rae=sum_absolute_diff/sum_absolute_r 
    rae=rae.item()
###########################################################################################################


    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g) #Pearson's correlation coefficient?
    correlation = (correlation[index]).mean()

    #s-mape
    smape=0
    for z in range(Ytest.shape[1]):
        smape+=s_mape(Ytest[:,z],predict[:,z])
    smape/=Ytest.shape[1]

    #plot random 4 columns
    counter=0
    if is_plot:
        for v in range(r,r+142):
            col=v%data.m
            

            #plot_predicted_actual(predict[:,col],Ytest[:,col],'Testing node '+str(col)+', RRSE= '+str(round(rrse,2))+', CORR= '+str(round(correlation,2)),counter)
            node_name=DataLoaderS.col[col].replace('-ALL','').replace('Mentions-','Mentions of ').replace(' ALL','').replace('Solution_','').replace('_Mentions','')
            node_name=consistent_name(node_name)
            
            #save error to file
            save_metrics_1d(torch.from_numpy(predict[:,col]),torch.from_numpy(Ytest[:,col]),node_name,'Testing')
            #plot
            plot_predicted_actual(predict[:,col],Ytest[:,col],node_name, 'Testing',variance[:,col],confidence_95[:,col])
            counter+=1

    return rrse,rae,correlation, smape

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, is_plot):
    #model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    variance=None
    confidence_95=None
    sum_squared_diff=0
    sum_absolute_diff=0
    r=0 #random.randint(0, 141)#random column
    print('validation r=',str(r))

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)

        # Bayesian estimation
        num_runs = runs

        # Create a list to store the outputs
        outputs = []

        # Run the model multiple times
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(X)
                output = torch.squeeze(output)
                if len(output.shape) == 1 or len(output.shape) == 2:
                    output = output.unsqueeze(dim=0)
                outputs.append(output)
            

        # Stack the outputs along a new dimension
        outputs = torch.stack(outputs)

        # Calculate mean, variance, and standard deviation
        mean = torch.mean(outputs, dim=0)
        var = torch.var(outputs, dim=0)#variance
        std_dev = torch.std(outputs, dim=0)#standard deviation

        # Calculate 95% confidence interval
        z=1.96
        confidence=z*std_dev/torch.sqrt(torch.tensor(num_runs))

        # Print the results
        # print("Mean:", mean)
        # print("Variance:", variance)
        # print("Standard Deviation:", std_dev)

        output=mean #we will consider the mean to be the prediction

        scale = data.scale.expand(Y.size(0), Y.size(1), data.m) #scale will have the max of each column (142 max values)
        
        #inverse normalisation
        output*=scale
        Y*=scale
        var*=scale
        confidence*=scale

        # #inverse difference
        # output=inverse_diff_3d(output,I,data.shift)
        # Y=inverse_diff_3d(Y,I,data.shift)

        if predict is None:
            predict = output
            test = Y
            variance=var
            confidence_95=confidence
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))
            variance= torch.cat((variance, var))
            confidence_95=torch.cat((confidence_95,confidence))


        print('EVALUATE RESULTS:')
        scale = data.scale.expand(Y.size(0), Y.size(1), data.m) #scale will have the max of each column (142 max values)
        y_pred_o=output
        y_true_o=Y
        for z in range(Y.shape[1]):
            print(y_pred_o[0,z,r],y_true_o[0,z,r]) #only one col
        
        total_loss += evaluateL2(output, Y).item()
        total_loss_l1 += evaluateL1(output, Y).item()
        n_samples += (output.size(0) * output.size(1) * data.m)

        #RRSE according to Lai et.al
        sum_squared_diff += torch.sum(torch.pow(Y - output, 2))
        #Relative Absolute Error RAE 
        sum_absolute_diff+=torch.sum(torch.abs(Y - output))


    rse = math.sqrt(total_loss / n_samples) / data.rse #Root Relative Squared Error (RRSE)?
    rae = (total_loss_l1 / n_samples) / data.rae # Relative Absolute Error?

##########################################################################################################
    #RRSE according to Lai et.al
    root_sum_squared= math.sqrt(sum_squared_diff) #numerator
    
    # scale = data.scale.expand(test.size(0), test.size(1), data.m) #scale will have the max of each column (142 max values)
    # test_s= test*scale
    test_s=test
    mean_all = torch.mean(test_s, dim=(0,1)) # calculate the mean of each column in test call it Yj-
    diff_r = test_s - mean_all.expand(test_s.size(0), test_s.size(1), data.m) # subtract the mean from each element in the tensor test
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))# square the result and sum over all elements
    root_sum_squared_r=math.sqrt(sum_squared_r)#denominator

    #RRSE according to Lai et.al
    rrse=root_sum_squared/root_sum_squared_r

###########################################################################################################
###########################################################################################################

    #Relative Absolute Error RAE 
    sum_absolute_r=torch.sum(torch.abs(diff_r))# absolute the result and sum over all elements
    rae=sum_absolute_diff/sum_absolute_r 
    rae=rae.item()
###########################################################################################################


    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0)/ (sigma_p * sigma_g) #Pearson's correlation coefficient?
    correlation = (correlation[index]).mean()

    #s-mape
    smape=0
    for x in range(Ytest.shape[0]):
        for z in range(Ytest.shape[2]):
            smape+=s_mape(Ytest[x,:,z],predict[x,:,z])
    smape/=Ytest.shape[0]*Ytest.shape[2]


    #plot random 4 columns
    counter=0
    if is_plot:
        for v in range(r,r+142):
            col=v%data.m
            node_name=DataLoaderS.col[col].replace('-ALL','').replace('Mentions-','Mentions of ').replace(' ALL','').replace('Solution_','').replace('_Mentions','')
            node_name=consistent_name(node_name)
            save_metrics_1d(torch.from_numpy(predict[-1,:,col]),torch.from_numpy(Ytest[-1,:,col]),node_name,'Validation')
            plot_predicted_actual(predict[-1,:,col],Ytest[-1,:,col],node_name, 'Validation', variance[-1,:,col], confidence_95[-1,:,col])
            counter+=1
    return rrse, rae, correlation, smape

def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0

    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        #temp_X=X
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        if iter % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]

            id = torch.tensor(id).to(device)
            tx = X[:, :, :, :] #id was in third colum
            ty = Y[:, :, :] #id was in third colum
            
            # Bayesian estimation
            num_runs = runs

            # Create a list to store the outputs
            outputs = []

            # Run the model multiple times
            for _ in range(num_runs):
                output = model(tx)
                output = torch.squeeze(output,3)
                outputs.append(output)
            

            # Stack the outputs along a new dimension
            outputs = torch.stack(outputs)

            # print(outputs[0][0][0])
            # print(outputs[1][0][0])
            # sys.exit()

            # Calculate mean, variance, and standard deviation
            mean = torch.mean(outputs, dim=0)
            var = torch.var(outputs, dim=0)#variance
            std_dev = torch.std(outputs, dim=0)#standard deviation
            
            # Calculate 95% confidence interval
            z=1.96
            confidence=z*std_dev/torch.sqrt(torch.tensor(num_runs))

            output=mean #we will consider the mean to be the prediction
            
   
           
            scale = data.scale.expand(output.size(0), output.size(1), data.m)
            scale = scale[:,:,:] #id was in third colum
            
            output*=scale #by Zaid
            ty*=scale
            
            # #inverse diff
            # output=inverse_diff_3d(output,I,data.shift)
            # ty=inverse_diff_3d(ty,I,data.shift)

            loss = criterion(output, ty)
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * output.size(1) * data.m)
            
            # perform gradient clipping
            #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            grad_norm = optim.step()

        if iter%1==0:
            print('iter:{:3d} | loss: {:.3f}'.format(iter,loss.item()/(output.size(0) * output.size(1)* data.m)))
        iter += 1
    return total_loss / n_samples


parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='./data/sm_data.txt',
                    help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='modelb'+str(runs)+'.pt',
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device',type=str,default='cuda:1',help='')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=142,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=20,help='k')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=2,help='dilation exponential')
parser.add_argument('--conv_channels',type=int,default=16,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=16,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=32,help='skip channels')
parser.add_argument('--end_channels',type=int,default=64,help='end channels')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=10,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=36,help='output sequence length')
parser.add_argument('--horizon', type=int, default=1) #zaid modified it!!!!!
parser.add_argument('--layers',type=int,default=5,help='number of layers')

parser.add_argument('--batch_size',type=int,default=8,help='batch size')
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')

parser.add_argument('--clip',type=int,default=10,help='clip')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='tanh alpha')

parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--step_size',type=int,default=100,help='step_size')


args = parser.parse_args()
device = torch.device('cpu') #modified!!!!!!!!!!!!!!!!!!!!!
torch.set_num_threads(3)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fixed_seed = 123

def main(experiment):


    # gcn_depths=[1,2,3]
    # lrs=[0.001, 0.0001,0.005, 0.0005]
    # convs=[4,8,16,32]
    # ress=[4,8,16,32]
    # skips=[4,8,16,32,64]
    # ends=[16,32,64,128]
    # layers=[1,2,3]
    # ks=[10,20,30,40,50,60]
    # dropouts=[0.2,0.3,0.4,0.5]
    # dilation_exs=[1,2,3]
    # node_dims=[10,20,30,40,50,60]
    # prop_alphas=[0.05,0.1,0.15,0.2,0.3,0.4,0.6,0.8]
    # tanh_alphas=[0.05,0.1,0.5,1,2,3,5,7,9]

    gcn_depths=[1,2,3]
    lrs=[0.01,0.001,0.0005,0.0008,0.0001,0.0003,0.005]#[0.00001,0.0001,0.0002,0.0003]
    convs=[4,8,16]
    ress=[16,32,64]
    skips=[64,128,256]
    ends=[256,512,1024]
    layers=[1,2]
    ks=[20,30,40,50,60,70,80,90,100]
    dropouts=[0.2,0.3,0.4,0.5,0.6,0.7]
    dilation_exs=[1,2,3]
    node_dims=[20,30,40,50,60,70,80,90,100]
    prop_alphas=[0.05,0.1,0.15,0.2,0.3,0.4,0.6,0.8]
    tanh_alphas=[0.05,0.1,0.5,1,2,3,5,7,9]

    # gcn_depths=[3]
    # lrs=[0.005]
    # convs=[4]
    # ress=[8]
    # skips=[16]
    # ends=[32]
    # layers=[2]
    # ks=[80]
    # dropouts=[0.4]
    # dilation_exs=[3]
    # node_dims=[100]
    # prop_alphas=[0.05]
    # tanh_alphas=[3]

    best_val = 10000000
    best_rse=  10000000
    best_rae=  10000000
    best_corr= -10000000
    best_smape=10000000
    
    best_test_rse=10000000
    best_test_corr=-10000000

    best_hp=[]


    #random search
    for q in range(30):

        #hp
        gcn_depth=gcn_depths[randrange(len(gcn_depths))]
        lr=lrs[randrange(len(lrs))]
        conv=convs[randrange(len(convs))]
        res=ress[randrange(len(ress))]
        skip=skips[randrange(len(skips))]
        end=ends[randrange(len(ends))]
        layer=layers[randrange(len(layers))]
        k=ks[randrange(len(ks))]
        dropout=dropouts[randrange(len(dropouts))]
        dilation_ex=dilation_exs[randrange(len(dilation_exs))]
        node_dim=node_dims[randrange(len(node_dims))]
        prop_alpha=prop_alphas[randrange(len(prop_alphas))]
        tanh_alpha=tanh_alphas[randrange(len(tanh_alphas))]
        

        Data = DataLoaderS(args.data, 0.43, 0.30, device, args.horizon, args.seq_in_len, args.normalize,args.seq_out_len)
    
        # print(Data.shift)
        # for i in range(Data.dat.shape[1]):
        #     plot_data(Data.dat[:,i],'Node '+str(i)) 
        # sys.exit()     

        print('train X:',Data.train[0].shape)
        print('train Y:', Data.train[1].shape)
        print('valid X:',Data.valid[0].shape)
        print('valid Y:',Data.valid[1].shape)
        print('test X:',Data.test[0].shape)
        print('test Y:',Data.test[1].shape)
        print('test window:', Data.test_window.shape)

        print('length of training set=',Data.train[0].shape[0])#Zaid
        print('length of validation set=',Data.valid[0].shape[0])#Zaid
        print('length of testing set=',Data.test[0].shape[0])#Zaid
        print('valid=',int((0.43 + 0.3) * Data.n))
        
       
        
        model = gtnet(args.gcn_true, args.buildA_true, gcn_depth, args.num_nodes,
                    device, Data.adj, dropout=dropout, subgraph_size=k,
                    node_dim=node_dim, dilation_exponential=dilation_ex,
                    conv_channels=conv, residual_channels=res,
                    skip_channels=skip, end_channels= end,
                    seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                    layers=layer, propalpha=prop_alpha, tanhalpha=tanh_alpha, layer_norm_affline=False)
        
        # model = net_lstm(2, 40, args.gcn_true, args.buildA_true, gcn_depth, args.num_nodes,
        #                   device, Data.adj, None, dropout=dropout, subgraph_size=k, node_dim=node_dim,
        #                   seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len, layers=layer, propalpha=prop_alpha, tanhalpha=tanh_alpha,
        #                   layer_norm_affline=True)
        # model = model.to(device)


        print(args)
        print('The recpetive field size is', model.receptive_field)
        nParams = sum([p.nelement() for p in model.parameters()])
        print('Number of model parameters is', nParams, flush=True)

        if args.L1Loss:
            criterion = nn.L1Loss(reduction='sum').to(device)
        else:
            criterion = nn.MSELoss(reduction='sum').to(device)
        evaluateL2 = nn.MSELoss(reduction='sum').to(device) #MSE
        evaluateL1 = nn.L1Loss(reduction='sum').to(device) #MAE


        #best_val = 10000000
        optim = Optim(
            model.parameters(), args.optim, lr, args.clip, lr_decay=args.weight_decay
        )

        es_counter=0 #early stopping
        # At any point you can hit Ctrl + C to break out of training early.
        try:
            print('begin training')
            for epoch in range(1, args.epochs + 1):
                print('Experiment:',(experiment+1))
                print('Iter:',q)
                print('epoch:',epoch)
                print('hp=',[gcn_depth,lr,conv,res,skip,end, k, dropout, dilation_ex, node_dim, prop_alpha, tanh_alpha, layer, epoch])
                print('best sum=',best_val)
                print('best rrse=',best_rse)
                print('best rrae=',best_rae)
                print('best corr=',best_corr)
                print('best smape=',best_smape)       
                print('best hps=',best_hp)
                print('best test rse=',best_test_rse)
                print('best test corr=',best_test_corr)

                es_counter+=1
                # if(es_counter>30): #early stopping
                #     break

                epoch_start_time = time.time()
                train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
                val_loss, val_rae, val_corr, val_smape = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                                 args.batch_size,False)
                #val_loss, val_rae, val_corr, val_smape = evaluate_sliding_window(Data, Data.test_window, model, evaluateL2, evaluateL1,
                #                           args.seq_in_len, False)
                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f} | valid smape  {:5.4f}'.format(
                        epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr, val_smape), flush=True)
                # Save the model if the validation loss is the best we've seen so far.

                sum_loss=val_loss+val_rae-val_corr
                if (not math.isnan(val_corr)) and val_loss < best_rse:
                #if val_loss < best_rse:
                    with open(args.save, 'wb') as f:
                        torch.save(model, f)
                    best_val = sum_loss
                    best_rse= val_loss
                    best_rae= val_rae
                    best_corr= val_corr
                    best_smape=val_smape

                    best_hp=[gcn_depth,lr,conv,res,skip,end, k, dropout, dilation_ex, node_dim, prop_alpha, tanh_alpha, layer, epoch]
                    
                    es_counter=0

                    # test_acc, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                    #                                      args.batch_size)
                    
                    test_acc, test_rae, test_corr, test_smape = evaluate_sliding_window(Data, Data.test_window, model, evaluateL2, evaluateL1,
                                           args.seq_in_len, False)  #Zaid
                    print('********************************************************************************************************')
                    print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}| test smape {:5.4f}".format(test_acc, test_rae, test_corr, test_smape), flush=True)
                    print('********************************************************************************************************')
                    best_test_rse=test_acc
                    best_test_corr=test_corr

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    print('best val loss=',best_val)
    print('best hps=',best_hp)
    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    vtest_acc, vtest_rae, vtest_corr, vtest_smape = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                         args.batch_size, False)
    #vtest_acc, vtest_rae, vtest_corr, vtest_smape = evaluate_sliding_window(Data, Data.test_window, model, evaluateL2, evaluateL1,
    #                                    args.seq_in_len, True) 
    # test_acc, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
    #                                      args.batch_size)
    
    #test_acc, test_rae, test_corr, test_smape =0,0,0,0
    test_acc, test_rae, test_corr, test_smape = evaluate_sliding_window(Data, Data.test_window, model, evaluateL2, evaluateL1,
                                         args.seq_in_len, False) #Zaid
    print('********************************************************************************************************')    
    print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f} | test smape {:5.4f}".format(test_acc, test_rae, test_corr, test_smape))
    print('********************************************************************************************************')
    return vtest_acc, vtest_rae, vtest_corr, vtest_smape, test_acc, test_rae, test_corr, test_smape

if __name__ == "__main__":
    vacc = []
    vrae = []
    vcorr = []
    vsmape=[]
    acc = []
    rae = []
    corr = []
    smape=[]
    
    set_random_seed(fixed_seed)
    for i in range(5):
        val_acc, val_rae, val_corr, val_smape, test_acc, test_rae, test_corr, test_smape = main(i)
        vacc.append(val_acc)
        vrae.append(val_rae)
        vcorr.append(val_corr)
        vsmape.append(val_smape)
        acc.append(test_acc)
        rae.append(test_rae)
        corr.append(test_corr)
        smape.append(test_smape)
    print('\n\n')
    print('1 run average')
    print('\n\n')
    print("valid\trse\trae\tcorr\ts-mape")
    print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae), np.mean(vcorr), np.mean(vsmape)))
    print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae), np.std(vcorr), np.std(vsmape)))
    print('\n\n')
    print("test\trse\trae\tcorr\ts-mape")
    print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(acc), np.mean(rae), np.mean(corr), np.mean(smape)))
    print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(acc), np.std(rae), np.std(corr), np.std(smape)))

