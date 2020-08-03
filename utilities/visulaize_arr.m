%%
clear all;
format long;

set(0,'defaultAxesFontSize', 14);
set(0, 'DefaultLineLineWidth', 1);

%%
clear;
g_arr= readtable('../results/gpu_arr.csv');  % skips the first three rows of data

g = g_arr.array;
figure()
plot(0:length(g)-1, g);


