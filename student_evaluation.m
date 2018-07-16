%% Turkiye student Evaluation Dataset
% Unsupervised learning problem

%% Data Preprocessing
X = readtable('turkiye-student-evaluation_generic.csv');

% Check Missing values
sum(ismissing(X))

% there are no missing values in any column of this dataset.
% No need of data preprocessing

% All Attributes are numeric

% The nb.repeat is the target column of this dataset. It takes 3 values
% 1,2,and 3.

%%

% Target variable distribution
tabulate(X.nb_repeat)

% Split train and target data
Y=X.nb_repeat;
X.nb_repeat=[];
%% Exploratory Data Analysis 

for k=1:10
    colData = X.(k);
    figure(k)
    histogram(colData);
    xlabel(X.Properties.VariableNames(k));
    ylabel('Count');
end

figure(11)
histogram(Y);
xlabel('nbRepeat');
ylabel('Count');
%% Models with all data
rng('default') % For reproducibility

% Models 

% Naive Bayes
% K-nearest Neighbor
% Logistic Regression
% Decision Trees
% Random Forest
% Multi-Layer Perceptron

%% 
% <html><h3>Naive Bayes</h3></html>

Md1 = fitcnb(X,Y);
L1 = loss(Md1,X,Y);
AccuracyNB = 1-L1
%% 
% <html><h3>KNN</h3></html>
Md2 = fitcknn(X,Y,'NumNeighbors',4);
L2 = loss(Md2,X,Y);
AccuracyKNN = 1-L2
%% 
% <html><h3>Logistic Regression</h3></html>

[A,~] = mnrfit(table2array(X),Y);
yhat = mnrval(A,table2array(X));
[~,B]=max(yhat');
CP = classperf(Y,B')

%% 
% <html><h3>Random Forest</h3></html>

B = TreeBagger(50,X,Y,'OOBVarImp','On')
figure
plot(oobError(B))
xlabel('Number of Grown Trees')
ylabel('Out-of-Bag Classification Error')

mean(oobError(B))
%% 
% <html><h3>Decision Tree</h3></html>
Md3 = fitctree(X,Y,'MaxNumSplits',7,'CrossVal','on');
view(Md3.Trained{1},'Mode','graph')
numBranches = @(x)sum(x.IsBranch);
mdlDefaultNumSplits = cellfun(numBranches, Md3.Trained);

figure;
histogram(mdlDefaultNumSplits)

Error3 = kfoldLoss(Md3);
AccuracyDT=1-Error3
%% 
% <html><h3>Neural Network</h3></html>

x = table2array(X)';
t = full(ind2vec(Y'));

trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize, trainFcn);
net.input.processFcns = {'removeconstantrows','mapminmax'};

net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net.performFcn = 'crossentropy';  % Cross-Entropy

net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotconfusion', 'plotroc'};

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y)

% View the Network
% view(net)

% Plots
figure, plotperform(tr)
figure, plottrainstate(tr)
figure, ploterrhist(e)
figure, plotconfusion(t,y)
figure, plotroc(t,y)

%% Variables Distribution
variables=X.Properties.VariableNames; %get variable names
tot=zeros(5,1);
for k=6:length(variables)
    colData = X.(k);
    temp=tabulate(colData);
    tot = tot + temp(:,3);
end
tot=tot/(length(variables)-6)

% Looks like all variables have similar distribution

%% Models after removing question columns
% Lets remove unnecessary question columns as they have same distribution

% Models 

% Naive Bayes
% K-nearest Neighbor
% Logistic Regression
% Decision Trees
% Random Forest
% Multi-Layer Perceptron
%%
% <html><h3>Naive Bayes</h3></html>
mod_dat = X(:,1:4);
Md4 = fitcnb(mod_dat,Y);
L4 = loss(Md4,mod_dat,Y);
AccuracyNB=1-L4
%% 
% <html><h3>KNN</h3></html>
mod_dat = X(:,1:4);
Md5 = fitcknn(mod_dat,Y,'NumNeighbors',4);
L5 = loss(Md5,mod_dat,Y);
AccuracyKNN=1-L5
%% 
% <html><h3>Logistic Regression</h3></html>
[A,dev,stats] = mnrfit(table2array(mod_dat),Y);
yhat = mnrval(A,table2array(mod_dat));
[~,B]=max(yhat');
CP = classperf(Y,B')

%% 
% <html><h3>Decision Tree</h3></html>
Md6 = fitctree(mod_dat,Y,'MaxNumSplits',7,'CrossVal','on');
view(Md6.Trained{1},'Mode','graph');
numBranches = @(x)sum(x.IsBranch);
mdlDefaultNumSplits = cellfun(numBranches, Md6.Trained);

figure;
histogram(mdlDefaultNumSplits);

Error6 = kfoldLoss(Md6);
AccuracyDT=1-Error6
%% 
% <html><h3>Random Forest</h3></html>

B = TreeBagger(50,mod_dat,Y,'OOBVarImp','On');
figure
plot(oobError(B))
xlabel('Number of Grown Trees');
ylabel('Out-of-Bag Classification Error');

mean(oobError(B))

%% 
% <html><h3>Neural Network</h3></html>

x = table2array(mod_dat)';
t = full(ind2vec(Y'));

trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize, trainFcn);
net.input.processFcns = {'removeconstantrows','mapminmax'};

net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net.performFcn = 'crossentropy';  % Cross-Entropy

net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotconfusion', 'plotroc'};

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y)

% View the Network
% view(net)

% Plots
figure, plotperform(tr)
figure, plottrainstate(tr)
figure, ploterrhist(e)
figure, plotconfusion(t,y)
figure, plotroc(t,y)