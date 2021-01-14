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

Ts = 0.1;                 % Sampling time 
t_final = 10;             % Simulation end time
t = 0:Ts:t_final-Ts;      % Time axis
numIters = length(t);     % Number of iterations

x = [0;0];               % Initial State Vector
x_hat = x;               % Initial State Vector Estimate
P = [1 0;0 1];           % Initial Covariance Matrix
x_hat_previous = x;      % Previous State Vector
P_previous = P;          % Previous Covariance Matrix

A = [0.01   -0.4940 ; 0.1129 -0.3832];    % State Transition Matrix or Prediction Matrix
B = [0.5919 ; 0.5191];                    % Input Control Matrix
C = [1 0];                                % Measurement Matrix

u = sin(2*pi*0.025*t);      % Input signal or Control vector
Plant = ss(A,B,C,0,Ts);     % Plant
y = lsim(Plant,u,t)';       % Ideal measurement output
y_hat = zeros(1,length(y)); % Initial measurement output estimate

noise_v_pct = 10;           % Measurement Noise in Percent

sigma_w = 0.2;
mu_w = 0;
amp_w = 0.1;
noise_w = amp_w*sigma_w.*randn(1,length(t)) + mu_w;    % Process Noise
Q = sigma_w^2;                                         % Process Noise Covariance Matrix

sigma_v = 0.5;
mu_v = 0;
amp_v = noise_v_pct/100;
noise_v = amp_v*sigma_v.*randn(1,length(t)) + mu_v;    % Measurement Noise
R = sigma_v^2;                                         % Measurement Noise Covariance Matrix

Z = y + y.*noise_v;         % Measurement output with measurement noise

% Errors
measError = zeros(1,length(y));
estError = zeros(1,length(y));

% Distributions
pdf_time = -1:0.01:1-0.01;
timeStamp = 40;
pdf_current = zeros(length(pdf_time),length(t));
pdf_estimated = zeros(length(pdf_time),length(t));
pdf_measurement = zeros(length(pdf_time),length(t));

%% Predict-Update Process

for i=1:numIters
    % Predict
    [x_hat, P] = predict(x_hat_previous,P_previous,A,B,u(i),Q);
    
    % Update
    [x_hat_updated, P_updated] = update(x_hat,P,C,Z(i),R);
    y_hat(i) = C*x_hat_updated;
    
    x_hat_previous = x_hat_updated;
    P_previous = P_updated;

    % Errors
    measError(i) = y(i) - Z(i);
    estError(i) = y(i) - y_hat(i);
    
    % Distributions
    pdf_current(:,i) = normpdf(pdf_time,x_hat(1),P(1));
    pdf_measurement(:,i) = normpdf(pdf_time,Z(i),R);
    pdf_estimated(:,i) = normpdf(pdf_time,x_hat_updated(1),P_updated(1));
    
%     % Animated graph for Distributions
%     plot(pdf_time,pdf_current(:,i),'-g',pdf_time,pdf_measurement(:,i),'-r',pdf_time,pdf_estimated(:,i),'-b')
%     legend("Current State Estimate","Measured Output","Next Estimated State")
%     grid on;
%     hold off;
%     pause(0.1);
end

%% Plotting Kalman Filter outputs, Errors and Gaussian distributions

figure("Name","Kalman Filter with ideal measurement output, measured output and estimated output")
plot(t,y,'-b.',t,Z,'--.',t,y_hat,'-g.')
xlabel('Time (in sec)'),ylabel('Output values')
legend("Ideal Measurement Output","Measured Output","Estimated Output")
grid on;

figure("Name","Errors in Measured output and Estimated output")
plot(t,measError,'-r.',t,estError,'-g.')
xlabel('Time (in sec)'),ylabel('Error values')
legend("Measured Error","Estimated Error")
grid on;

figure("Name","Gaussian Distributions of State Estimates and Measurement")
plot(pdf_time,pdf_current(:,timeStamp),'-g',pdf_time,pdf_measurement(:,timeStamp),'-r',pdf_time,pdf_estimated(:,timeStamp),'-b')
legend("Current State Estimate","Measured Output","Next Estimated State")
grid on;

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