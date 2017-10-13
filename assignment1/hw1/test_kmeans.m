clear all;
close all;
clc;


mu1=[0 0 0];
S1=[0.3 0 0;0 0.35 0;0 0 0.3];
data1=mvnrnd(mu1,S1,100);


mu2=[1.25 1.25 1.25];
S2=[0.3 0 0;0 0.35 0;0 0 0.3];
data2=mvnrnd(mu2,S2,100);


mu3=[-1.25 1.25 -1.25];
S3=[0.3 0 0;0 0.35 0;0 0 0.3];
data3=mvnrnd(mu3,S3,100);


plot3(data1(:,1),data1(:,2),data1(:,3),'+');
hold on;
plot3(data2(:,1),data2(:,2),data2(:,3),'r+');
plot3(data3(:,1),data3(:,2),data3(:,3),'g+');
grid on;


data=[data1;data2;data3];


[u, c]=KMeans(data,3); 
[m, n]=size(data);


figure;
hold on;
for i=1:m 
    if c(i) == 1   
         plot3(data(i,1), data(i,2), data(i,3),'ro'); 
    elseif c(i) == 2
         plot3(data(i,1), data(i,2), data(i,3),'go'); 
    else 
         plot3(data(i,1), data(i,2), data(i,3),'bo'); 
    end
end
grid on;