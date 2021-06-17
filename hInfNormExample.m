clear all; close all; clc;
F = [-0.8986, 0.9508; -0.05614, 0.5699];
H = [0.4435, 0.8629; -0.2202, -0.1005];

K = [-0.3753, -0.4155; -0.0937, 0.0631];

tildeF = F - K*H';

sys = ss(tildeF, K*H', eye(2), zeros(2), -1);
ninf = (hinfnorm(sys))^2
norm(sys,inf)^2

