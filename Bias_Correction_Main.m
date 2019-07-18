clc
close all
clear all
tic
Simulation_State=1;  % 1: Without bias correction  2: Cole's method   3: Proposed Method

if Simulation_State==1
    disp('Without bias correction:')
elseif Simulation_State==2
    
    disp('Cole''s method:')
elseif Simulation_State==3
    disp('Proposed Method:')
end

load Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Data sets (Training and Test )  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Training Set , 675 HC
HC_DATA_Train=Data.train.features;
HC_Age_Train=Data.train.age;

%% Test sets
% 1: indendent 75 HC
HC_DATA_Test=Data.Test.HC.Features;
HC_Age_Test=Data.Test.HC.Age;
% 2 : MCI
MCI_DATA=Data.Test.MCI.Features;
MCI_Age=Data.Test.MCI.Age;
%[row, col] = find(isnan(MCI_Age));
%MCI_DATA(row,:)=[];
%MCI_Age(row,:)=[];
%  3 : AD
AD_DATA=Data.Test.AD.Features;
AD_Age=Data.Test.AD.Age;
%[row, col] = find(isnan(AD_Age));
%AD_DATA(row,:)=[];
%AD_Age(row,:)=[];
%%%%%%%#######################################################################################################

[ n , m ] = size (HC_DATA_Train);
PredictTest = zeros(size(HC_Age_Train));

K  = 10 ; % nomber of folds
Foldaccuracy= zeros(1,K);
bestacctrainmain=zeros(1,K);
cvFolds =  zeros(n,1);

for z = 1 :K : n
    for w = 1 : K
        cvFolds(z) = w;
        z = z+1;
        % w = w+1 ;
    end
end
cvFolds = cvFolds( 1:n ,:);


for  i =1:K                                 %for each fold  for i = 1 : K
    testIdx = (cvFolds == i);                % get indices of test instances
    trainIdx = ~testIdx;                     % get indices training instances
    %%
    AgeTrain=HC_Age_Train(trainIdx,:);
    MainTestAge=HC_Age_Train(testIdx,:);
    DataTrain=HC_DATA_Train(trainIdx,:);
    MainTestData=HC_DATA_Train(testIdx,:);
    
    
    %% Regression Model
    Mdl = fitrsvm(DataTrain,AgeTrain,'KernelFunction','linear');
    XX= predict(Mdl,MainTestData);
    PredictTest_Before(testIdx,1)=XX;
    
    
    %         if Simulation_State==2
    %             p = polyfit(MainTestAge, (XX),1);
    %             q(i)=p(1);
    %             qq(i)=p(2);
    %         elseif Simulation_State==3
    %
    %             p = polyfit(MainTestAge, (XX-MainTestAge),1);
    %             q(i)=p(1);
    %             qq(i)=p(2);
    %         end
    
    
end


if Simulation_State==2
    p = polyfit(HC_Age_Train, (PredictTest_Before),1);
    q=p(1);
    qq=p(2);
elseif Simulation_State==3
    
    p = polyfit(HC_Age_Train, (PredictTest_Before-HC_Age_Train),1);
    q=p(1);
    qq=p(2);
end

PredictTest=[];
if Simulation_State==1
    PredictTest=PredictTest_Before;
elseif Simulation_State==2
    
    PredictTest=(PredictTest_Before - mean(qq))./mean(q);
elseif Simulation_State==3
    
    Offset=mean(q).*HC_Age_Train+mean(qq);
    for t=1:size(PredictTest_Before,1)
        PredictTest(t,1)=PredictTest_Before(t,1)-Offset(t,1);
    end
end



MAEtest(1,1)=sum(abs(PredictTest -HC_Age_Train))/numel(HC_Age_Train);
RMSEtest(1,1)= (mean((PredictTest -HC_Age_Train).^2))^0.5;
MEANHCs(1,1)=mean((PredictTest -HC_Age_Train));
[RTest, Pvalue] = corr(HC_Age_Train,PredictTest);
R2_Train(1,1)=RTest.*RTest;


subplot(2,2,1);plot( HC_Age_Train,PredictTest-HC_Age_Train, 'go','MarkerSize',8 )
xlabel('Real age (years)','FontSize', 20)
ylabel('Delta brain age (years)','FontSize', 20)
coeff = polyfit(HC_Age_Train,PredictTest-HC_Age_Train,1);
xline = linspace( min(min(HC_Age_Train)), max(max(HC_Age_Train)), 2000);
yline = coeff(1)*xline+coeff(2);
hold on
plot(xline,yline,'g-')
hold on


[p,S] = polyfit(HC_Age_Train,PredictTest-HC_Age_Train,1);
[y_fit,delta] = polyval(p,HC_Age_Train,S);
%plot(Age_HC2,PredictTest,'bo')
hold on
plot(HC_Age_Train,y_fit,'g-')
%grid on
grid minor
subplot(2,2,1);plot(HC_Age_Train,y_fit+2*delta,'g--',HC_Age_Train,y_fit-2*delta,'g--','LineWidth',2)
title('Linear Fit of Data with 95% Prediction Interval, Training set : HC','FontSize', 15)
%legend('Data','Linear Fit','95% Prediction Interval')





%%############################################## . ON INDEPENDENT TEST SETS #######################################

%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$%1 :INDEPENDENT HC

Mdl_onTest_HC = fitrsvm(HC_DATA_Train,HC_Age_Train,'KernelFunction','linear');
PredictTest_HC_Before= predict(Mdl_onTest_HC,HC_DATA_Test);

PredictTest_HC=[];

if Simulation_State==1
    PredictTest_HC=PredictTest_HC_Before;
elseif Simulation_State==2
    
    PredictTest_HC=(PredictTest_HC_Before - mean(qq))./mean(q);
elseif Simulation_State==3
    
    Offset=mean(q).*HC_Age_Test+mean(qq);
    for t=1:size(PredictTest_HC_Before,1)
        PredictTest_HC(t,1)=PredictTest_HC_Before(t,1)-Offset(t,1);
    end
    
end




Mean_HC_Final(1,1)=mean(PredictTest_HC-HC_Age_Test);
MAE_HC_Final(1,1)=sum(abs(PredictTest_HC-HC_Age_Test))/numel(HC_Age_Test);
RMSE_HC_Final(1,:)= (mean((PredictTest_HC-HC_Age_Test).^2))^0.5;
[R_HC_Final, Pvalue_HC_Final] = corr(PredictTest_HC,HC_Age_Test);
R2_HC_Final(1,:)=R_HC_Final.*R_HC_Final;
% hold on
subplot(2,2,2);plot( HC_Age_Test,PredictTest_HC-HC_Age_Test, 'bo' )
xlabel('Real age (years)','FontSize', 20)
ylabel('Delta brain age (years)','FontSize', 20)
coeff = polyfit(HC_Age_Test,PredictTest_HC-HC_Age_Test,1);
xline = linspace( min(min(HC_Age_Test)), max(max(HC_Age_Test)), 2000);
yline = coeff(1)*xline+coeff(2);
hold on
plot(xline,yline,'b-')
hold on




[p,S] = polyfit(HC_Age_Test,PredictTest_HC-HC_Age_Test,1);
[y_fit,delta] = polyval(p,HC_Age_Test,S);
hold on
plot(HC_Age_Test,y_fit,'b-')
grid minor
subplot(2,2,2);plot(HC_Age_Test,y_fit+2*delta,'b--',HC_Age_Test,y_fit-2*delta,'b--','LineWidth',2)
title('Linear Fit of Data with 95% Prediction Interval, Independent Test set: HC ','FontSize', 15)
hold on
subplot(2,2,3);plot( HC_Age_Test,PredictTest_HC-HC_Age_Test, 'bo' )
subplot(2,2,3);plot(xline,yline,'b-')




%%%%%%%%%%%#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$%%%%%%%%%%%%%%%%%%%%%#######%%%% . TEST ON MCI

Mdl_onTest_MCI = fitrsvm(HC_DATA_Train,HC_Age_Train,'KernelFunction','linear');
PredictTest_MCI_Before= predict(Mdl_onTest_MCI,MCI_DATA);

PredictTest_MCI=[];

if Simulation_State==1
    PredictTest_MCI=PredictTest_MCI_Before;
elseif Simulation_State==2
    PredictTest_MCI=(PredictTest_MCI_Before - mean(qq))./mean(q);
elseif Simulation_State==3
    Offset=mean(q).*MCI_Age+mean(qq);
    for t=1:size(PredictTest_MCI_Before,1)
        PredictTest_MCI(t,1)=PredictTest_MCI_Before(t,1)-Offset(t,1);
    end
end




Mean_MCI_Final(1,1)=mean(PredictTest_MCI-MCI_Age);
MAE_MCI_Final(1,1)=sum(abs(PredictTest_MCI-MCI_Age))/numel(MCI_Age);
RMSE_MCI_Final(1,:)= (mean((PredictTest_MCI-MCI_Age).^2))^0.5;
[R_MCI_Final, Pvalue_MCI_Final] = corr(PredictTest_MCI,MCI_Age);
R2_MCI_Final(1,:)=R_MCI_Final.*R_MCI_Final;
hold on
subplot(2,2,3);plot( MCI_Age,PredictTest_MCI-MCI_Age, 'ko' )
title('Linear Fit of Data with 95% Prediction Interval, Independent Test sets: MCI & AD ','FontSize', 15)


xlabel('Real age (years)','FontSize', 20)
ylabel('Delta brain age (years)','FontSize', 20)
coeff = polyfit(MCI_Age,PredictTest_MCI-MCI_Age,1);
xline = linspace( min(min(MCI_Age)), max(max(MCI_Age)), 2000);
yline = coeff(1)*xline+coeff(2);
grid on
hold on
plot(xline,yline,'k-')
hold on


%%$$$$$$$$$$$$$$$$$$$$$$$$$$$%%%%%%%%%%@#######################################@@@@ 3 AD

Mdl_onTest_AD = fitrsvm(HC_DATA_Train,HC_Age_Train,'KernelFunction','linear');
PredictTest_AD_Before= predict(Mdl_onTest_AD,AD_DATA);
PredictTest_AD=[];

if Simulation_State==1
    PredictTest_AD=PredictTest_AD_Before;
elseif Simulation_State==2
    PredictTest_AD=(PredictTest_AD_Before - mean(qq))./mean(q);
elseif Simulation_State==3
    Offset=mean(q).*AD_Age+mean(qq);
    for t=1:size(PredictTest_AD_Before,1)
        PredictTest_AD(t,1)=PredictTest_AD_Before(t,1)-Offset(t,1);
    end
end



Mean_AD_Final(1,1)=mean(PredictTest_AD-AD_Age);
MAE_AD_Final(1,1)=sum(abs(PredictTest_AD-AD_Age))/numel(AD_Age);
RMSE_AD_Final(1,:)= (mean((PredictTest_AD-AD_Age).^2))^0.5;
[R_AD_Final, Pvalue_AD_Final] = corr(PredictTest_AD,AD_Age);
R2_AD_Final(1,:)=R_AD_Final.*R_AD_Final;
hold on
subplot(2,2,3);plot( AD_Age,PredictTest_AD-AD_Age, 'ro' )

xlabel('Real age (years)','FontSize', 20)
ylabel('Delta brain age (years)','FontSize', 20)
coeff = polyfit(AD_Age,PredictTest_AD-AD_Age,1);
xline = linspace( min(min(AD_Age)), max(max(AD_Age)), 2000);
yline = coeff(1)*xline+coeff(2);
hold on
plot(xline,yline,'r-')
hold on





Group = {'Training set : HC';'Test set : HC';'Test set : MCI';'Test set : AD'};
MAE = [MAEtest;MAE_HC_Final;MAE_MCI_Final;MAE_AD_Final];
RMSE = [RMSEtest;RMSE_HC_Final;RMSE_MCI_Final;RMSE_AD_Final];
R2 = [R2_Train;R2_HC_Final;R2_MCI_Final;R2_AD_Final];
Mean_Delta_Age = [MEANHCs;Mean_HC_Final;Mean_MCI_Final;Mean_AD_Final];
Results = table(Group,MAE,RMSE,R2,Mean_Delta_Age)

toc