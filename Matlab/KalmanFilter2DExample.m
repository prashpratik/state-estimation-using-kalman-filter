%% Kalman Filter Case Study
% x_hat(k) = A*x_hat_previous(k-1) + B*u(k)
% In a Kalman filter problem the known parameters are the input excitation
% u(k), the measured output y(k) and the computation system estimates
% the state of the system x_hat(k). x_hat(k) is linearly related to the
% output estimate as y_hat(k) = C*x_hat(k). The state parameter matrices
% A, B, C are assumed to be known via empirical means also known as system
% model.

clc;
clear all;
close all;

%% Setting up the known parameters x,x_hat,P,A,B,C,u(k),y(k),Z,Q,R

Ts = 0.5;                 % Sampling time 
t_final = 20;             % Simulation end time
t = 0:Ts:t_final-Ts;      % Time axis
numIters = length(t);     % Number of iterations

xX = [0;10];               % Initial State Vector
x_hatX = xX;               % Initial State Vector Estimate
PX = [1 0;0 1];            % Initial Covariance Matrix
x_hat_previousX = xX;      % Previous State Vector
P_previousX = PX;          % Previous Covariance Matrix

xY = [0;10];               % Initial State Vector
x_hatY = xY;               % Initial State Vector Estimate
PY = [1 0;0 1];            % Initial Covariance Matrix
x_hat_previousY = xY;      % Previous State Vector
P_previousY = PY;          % Previous Covariance Matrix

A = [1   Ts ; 0 1];      % State Transition Matrix or Prediction Matrix
B = [Ts^2/2 ; Ts];       % Input Control Matrix
C = [1 0];               % Measurement Matrix

uX = zeros(1,length(t));       % Input signal or Control vector
uY = -ones(1,length(t));       % Input signal or Control vector

yX = zeros(1,length(t));       % Initial Ideal Measurement output
yY = zeros(1,length(t));       % Initial Ideal Measurement output

y_hatX = zeros(1,length(t));   % Initial Estimated output
y_hatY = zeros(1,length(t));   % Initial Estimated output

noise_v_pct = 5;               % Measurement Noise in Percent

xXactual = xX;
xYactual = xY;
for i=1:numIters  
    yX(i) = C*xXactual;                 % Ideal Measurement output
    xXactual = A*xXactual + B*uX(i);
    
    yY(i) = C*xYactual;                 % Ideal Measurement output
    xYactual = A*xYactual + B*uY(i);
end

sigma_w = 0.1;
mu_w = 0;
amp_w = 0.05;
noise_w = amp_w*sigma_w.*randn(1,length(t)) + mu_w;    % Process Noise
Q = sigma_w^2;                                         % Process Noise Covariance Matrix

sigma_v = 0.8;
mu_v = 0;
amp_v = noise_v_pct/100;
noise_v = amp_v*sigma_v.*randn(1,length(t)) + mu_v;    % Measurement Noise
R = sigma_v^2;                                         % Measurement Noise Covariance Matrix

ZX = yX + yX.*noise_v;         % Measurement output with measurement noise

ZY = yY + yY.*noise_v;         % Measurement output with measurement noise

%% Predict-Update Process

for i=1:numIters
    % Predict
    [x_hatX, PX] = predict(x_hat_previousX,P_previousX,A,B,uX(i),Q);
    [x_hatY, PY] = predict(x_hat_previousY,P_previousY,A,B,uY(i),Q);
    
    % Update
    [x_hat_updatedX, P_updatedX] = update(x_hatX,PX,C,ZX(i),R);
    y_hatX(i) = C*x_hat_updatedX;
    
    x_hat_previousX = x_hat_updatedX;
    P_previousX = P_updatedX;
    
    [x_hat_updatedY, P_updatedY] = update(x_hatY,PY,C,ZY(i),R);
    y_hatY(i) = C*x_hat_updatedY;
    
    x_hat_previousY = x_hat_updatedY;
    P_previousY = P_updatedY;
end

%% Plotting ideal measurement output, measured output and estimated output

figure("Name","Kalman Filter with ideal measurement output, measured output and estimated output")

for i=1:numIters
    if i==1
        yXplot = yX(i);
        yYplot = yY(i);
        ZXplot = ZX(i);
        ZYplot = ZY(i);
        y_hatXplot = y_hatX(i);
        y_hatYplot = y_hatY(i);
    else
        yXplot = [yXplot yX(i)];
        yYplot = [yYplot yY(i)];
        ZXplot = [ZXplot ZX(i)];
        ZYplot = [ZYplot ZY(i)];
        y_hatXplot = [y_hatXplot y_hatX(i)];
        y_hatYplot = [y_hatYplot y_hatY(i)];
    end
    plot(yXplot,yYplot,'-b.',ZXplot,ZYplot,'--.',y_hatXplot,y_hatYplot,'-g.')
    xlabel('X-Position (in m)'),ylabel('Y-Position (in m)')
    xlim([min(min(yX),min(min(ZX),min(y_hatX))) max(max(yX),max(max(ZX),max(y_hatX)))+max(yX)/4])
    ylim([min(min(yY),min(min(ZY),min(y_hatY))) max(max(yY),max(max(ZY),max(y_hatY)))+max(yY)/4])
    legend("Ideal Measurement Output","Measured Output","Estimated Output")
    grid on;
    pause(0.1);
end

%% Predict Function

function [newStateVector, newCovariance] = predict(previousStateVector,previousCovariance,stateTransitionMatrix,controlMatrix,controlVector,processNoiseCovariance)
    x_hat_previous = previousStateVector;
    P_previous = previousCovariance;
    A = stateTransitionMatrix;
    B = controlMatrix;
    u = controlVector;
    Q = processNoiseCovariance;
    
    x_hat = A*x_hat_previous + B*u;
    P = A*P_previous*A' + Q;
    
    newStateVector = x_hat;
    newCovariance = P;
end

%% Update Function

function [updatedStateVector, updatedCovariance] = update(StateVector,Covariance,measurementMatrix,measurementOutput,measurementNoiseCovariance)
    x_hat = StateVector;
    P = Covariance;
    C = measurementMatrix;
    Z = measurementOutput;
    R = measurementNoiseCovariance;

    K = P*C'/((C*P*C') + R);  % Kalman Gain
    x_hat_updated = x_hat + K * (Z - (C*x_hat));
    P_updated = (eye(size(C)) - (K*C)) * P;
    
    updatedStateVector = x_hat_updated;
    updatedCovariance = P_updated;
end