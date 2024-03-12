from layer import *
import sys
import time
from util import DataLoaderS
import random
from matplotlib import pyplot
import random
import seaborn as sns
import numpy
from mpl_toolkits.mplot3d import Axes3D


fixed_seed=123



class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(gtnet, self).__init__()
        
        self.attention_scores = None  # Initialize attention_scores
        self.adp=None
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)


        #self.graph_attention = GraphAttentionLayer(residual_channels, out_channels=residual_channels, num_heads=1)

        self.idx = torch.arange(self.num_nodes).to(device)
        


    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))



        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    self.adp = self.gc(self.idx) # this line computes the adjacency matrix adaptively by calling the function forward in the gc
                else:
                    self.adp = self.gc(idx)
            else:
                self.adp = self.predefined_A
        


        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))

        for i in range(self.layers):
            residual = x


            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training) 
                    
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x1, attention_matrix1 = self.gconv1[i](x, self.adp)
                x2, attention_matrix2 = self.gconv2[i](x, self.adp.transpose(1, 0))
                x = x1 + x2
                attention_matrix = (attention_matrix1 + attention_matrix2) / 2  # Combine attention matrices
                self.attention_scores = attention_matrix  # Save attention scores
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)


        # Apply the attention layer
        #x, attention_matrix = self.graph_attention(x, self.adp)
        #x = x.view(8, 32, 142, 4)  # Adjust the dimensions accordingly


        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

       # print('attention_matrix:')
       # print(attention_matrix)
        
        return x

    # def compute_saliency(self, input):
    #     self.eval()  # Set the model to evaluation mode
    #     input.requires_grad = True  # Enable gradient computation for the input
    #     output = self(input)
    #     output.backward()  # Compute gradients

    #     saliency = input.grad.abs().max(dim=1)[0]  # Calculate the saliency map
    #     return saliency

    #Explainability
    # def compute_saliency(self, input, target=None):
    #     # Enable gradient computation for the input
    #     input.requires_grad = True

    #     # Forward pass
    #     output = self(input)

    #     gradient=output

    #     saliency = gradient.abs().max(dim=1)[0]  # Calculate the saliency map
    #     return saliency
    
    def compute_saliency(self, input, time_step, node_idx, time=False):

        # Enable gradient computation for the input
        input.requires_grad = True

        # Forward pass
        output = self(input)
        output = output[0,:,:,0]
        print(output.shape)


        # Compute the gradient of one time-step and one feature in the output
        # with respect to the input
        output[time_step,node_idx].backward()  # Assuming a shape like (sequence_length, features)
        print('output[',time_step,',',node_idx,']=',output[time_step,node_idx])


        # Get the gradients of the input
        gradient = input.grad.clone()
        # Make sure to zero out the gradients afterward if you plan to reuse the input
        input.grad.zero_()

        print('gradient=',gradient)

        index=-1
        if time:
            index=-2
        # Calculate the saliency map
        saliency = gradient.abs().max(dim=index)[0]
        print('saliency in gnet')
        print(saliency)
        return saliency



    #Explainability
    def explain_by_adjacency(self,node_idx, input):
        E={}
        col=DataLoaderS.col
        print('connections to node '+col[node_idx]+':',end='')
        counter=0
        for j in range(self.adp.shape[1]):
            if counter==3:
                break
            counter+=1
            if self.adp[node_idx,j].item()>0:
                # this will allow us to have adjacent node, adjacency weight, and adjacent node value in the input
                E[col[j]]= [self.adp[node_idx,j].item(), input[j].item()]
        print(E)        
        print(len(E))       
 
    def explain_by_adjacency(self):
        col=DataLoaderS.col
        for i in [0,1,2,3,4,5,100,101,102,103,104,105]: #subgraph
            print(col[i]+'-->',end='')
            for j in [0,1,2,3,4,5,100,101,102,103,104,105]:
                print(col[j]+':'+str(self.adp[i,j].item()),end=' ')
            print('')


    # Define the visualize_attention_scores function
    def visualize_attention_scores(self):
        # Plot attention scores
        attention_scores=self.attention_scores[0:10,0:10]
        pyplot.figure(figsize=(10, 8))
        sns.heatmap(attention_scores.cpu().numpy(), cmap="viridis", annot=True, fmt=".2f")
        pyplot.title("Attention Scores")
        pyplot.xlabel("Nodes")
        pyplot.ylabel("Nodes")
        pyplot.show()

    def visualize_attention_scores(self, names, rows,columns,title):
        # Plot attention scores
        attention_scores=self.attention_scores
        attention_scores = attention_scores[rows][:, columns]
        attention_scores/=torch.max(attention_scores)

        col_labels1 = [names[i] for i in rows] 
        col_labels2 = [names[j] for j in columns]


        print('max attention=',torch.max(attention_scores).item())


        # Check if col_labels length matches the subgraph size
        if len(col_labels1) != attention_scores.shape[0]:
            raise ValueError("Length of col_labels should match the subgraph size")

        #cmap = sns.light_palette("purple", as_cmap=True)
        if len(columns)>1:
            if len(columns)>30 and len(rows)<30:
                pyplot.figure(figsize=(20, 8))
            elif len(columns)>30 and len(rows)>30:
                pyplot.figure(figsize=(20, 20))
        else:
            pyplot.figure(figsize=(1, 8))



        heatmap=sns.heatmap(attention_scores.cpu().numpy(), cmap='OrRd', annot=False,
                    xticklabels=col_labels2, yticklabels=col_labels1)
        pyplot.title("Attention Scores")
        if len(columns)>30:
            heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90, fontsize=10)
        elif len(columns)==1:
            heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=0, fontsize=10)



        
        # pyplot.xlabel("Nodes")
        # pyplot.ylabel("Nodes")
        pyplot.savefig('Attention_'+title+'.pdf', format='pdf', bbox_inches="tight")
        pyplot.show()


    def visualize_filters(self):
        # Get the weights of the convolutional layers
        conv_weights = []
        for module in self.gate_convs[0].tconv:
            if isinstance(module, nn.Conv2d):
                conv_weights.append(module.weight.data)

        # Visualize the filters
        for i, weights in enumerate(conv_weights):
            num_filters = weights.size(0)
            fig, axes = pyplot.subplots(1, num_filters, figsize=(15, 3))  # Adjust figsize as needed

            # Plot the heatmaps
            for j in range(num_filters):
                heatmap = axes[j].imshow(weights[j, 0].cpu().numpy(), cmap='viridis', aspect='auto', interpolation='nearest')

                # Set title and turn off axis
                axes[j].set_title(f'Filter {j+1}')
                axes[j].axis('off')

            # Add a single colorbar for all the filters in the layer
            cbar = fig.colorbar(heatmap, ax=axes, orientation='vertical')
            cbar.set_label('Weight Values', rotation=270, labelpad=15)

            fig.suptitle(f'Layer {i+1} Filters')
            pyplot.show()




    def visualise_data(self,t1, t2, label1, label2):
        # Create a list of month-year labels starting from Mar-22 to Dec-25
        month_year_labels = ['Mar-22','Apr-22','May-22','Jun-22','Jul-22','Aug-22','Sep-22','Oct-22','Nov-22','Dec-22','Jan-23','Feb-23','Mar-23','Apr-23','May-23','Jun-23','July-23','Aug-23','Sep-23','Oct-23','Nov-23','Dec-23','Jan-24','Feb-24','Mar-24','Apr-24',
                             'May-24','Jun-24','July-24','Aug-24','Sep-24','Oct-24','Nov-24','Dec-24','Jan-25','Feb-25','Mar-25','Apr-25','May-25','Jun-25','July-25','Aug-25','Sep-25','Oct-25','Nov-25','Dec-25']

        # Set up the plot
        pyplot.figure(figsize=(10, 6))

        # Plot heatmap for tensor t1
        pyplot.subplot(2, 1, 1)
        sns.heatmap(t1.detach().unsqueeze(0).numpy(), cmap='YlGnBu', annot=False, fmt=".2f", xticklabels=month_year_labels,yticklabels=False)
        pyplot.axvline(x=10, color='red', linestyle='-', linewidth=2)       
        pyplot.title(f"{label1}")

        # Plot heatmap for tensor t2
        pyplot.subplot(2, 1, 2)
        sns.heatmap(t2.detach().unsqueeze(0).numpy(), cmap='YlGnBu', annot=False, fmt=".2f", xticklabels=month_year_labels,yticklabels=False)
        pyplot.axvline(x=10, color='red', linestyle='-', linewidth=2)
        pyplot.title(f"{label2}")

        # Show the plot
        pyplot.tight_layout()
        pyplot.savefig('Data_'+label1+'_'+label2+'.pdf', format='pdf', bbox_inches="tight")
        pyplot.show()







    # def visualize_attention_scores_3d(self, col_labels1, col_labels2, from1, to1, from2, to2):
    #     # Plot attention scores
    #     attention_scores = self.attention_scores
    #     attention_scores = attention_scores[from1:to1, from2:to2]
    #     attention_scores /= torch.max(attention_scores)

    #     print('max attention=', torch.max(attention_scores).item())

    #     # Check if col_labels length matches the subgraph size
    #     if len(col_labels1) != attention_scores.shape[0]:
    #         raise ValueError("Length of col_labels should match the subgraph size")

    #     fig = pyplot.figure(figsize=(12, 8))
    #     ax = fig.add_subplot(111, projection='3d')

    #     x, y = torch.meshgrid(torch.arange(attention_scores.shape[1]), torch.arange(attention_scores.shape[0]))
    #     x = x.reshape(-1).cpu().numpy()
    #     y = y.reshape(-1).cpu().numpy()
    #     z = attention_scores.reshape(-1).cpu().numpy()

    #     ax.bar3d(x, y, 0, 1, 1, z, shade=True, cmap='viridis')

    #     ax.set_xticks(torch.arange(attention_scores.shape[1]) + 0.5)
    #     ax.set_yticks(torch.arange(attention_scores.shape[0]) + 0.5)
    #     ax.set_xticklabels(col_labels2)
    #     ax.set_yticklabels(col_labels1)

    #     ax.set_title("Attention Scores (3D-like)")
    #     ax.set_xlabel("Nodes (Columns)")
    #     ax.set_ylabel("Nodes (Rows)")
    #     ax.set_zlabel("Attention Scores")

    #     pyplot.savefig('Attention_3D_like_' + str(from1) + '_' + str(to1) + '_' + str(from2) + '_' + str(to2) + '.pdf',
    #                 format='pdf', bbox_inches="tight")
    #     pyplot.show()





    # print('Forward...')
    # time.sleep(1)
    # print(adp[4])
    # time.sleep(3)
    # #sys.exit()

    # col=DataLoaderS.col
    # for i in range(adp.shape[0]):
    #     print('connections to node '+col[i]+': [',end='')
    #     counter=0
    #     for j in range(adp.shape[1]):
    #         if adp[i,j].item()>0:
    #             print(col[j],end='')
    #             if j<adp.shape[1]-1:
    #                 print(', ', end='')
    #             counter+=1
    #         if j==adp.shape[1]-1:
    #             print('] total=',counter)
    # sys.exit()

