%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  GARCH(1,1) volatility
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Import data 
load stocks
ret = diff(log(Fiat));  
dates = dates_stocks(2:end);  
prices = Fiat(2:end);    % first date and price discarded

save temp ret dates prices
clear
load temp
delete temp.mat


% Estimation the parameters of a normal Garch model getting the long run
% volatility and the kurtosis of the model

t0 = find(dates == datenum(2007,01,01));
T = length(dates);

ret_in = ret(1:t0-1);    % in-sample returns (until t0-1 included)
                         % note: t0-1 is 29/12/06
ret_out = ret(t0:end);   % out-of-sample returns (from t0 on)

model = garch(1,1);      % create a 'garch' object of type GARCH(1,1) 
                         % with normal innovations (by default)

model = estimate(model,ret_in);   % estimate the model on window 'ret_in'
                         % the output is the same 'model' as in input
                         % PLUS estimated parameters

omega = model.Constant;  % parameters are fields in the garch object
                         % omega corresponds to 'Constant' field
alpha = model.ARCH{1};   % alpha corresponds to 'ARCH' field
                         % 'ARCH' field is itself a cell array: use {1} to
                         % access its (unique) entry
beta = model.GARCH{1};   % beta corresponds to 'GARCH' field

vol_lr = sqrt(omega/(1-alpha-beta));  
                         % long-run (i.e. unconditional) volatility
kurt_lr = 3*(1-(alpha+beta)^2)/(1-(alpha+beta)^2-2*alpha^2);  
                         % long-run kurtosis (check slides)
disp([vol_lr kurt_lr])


% Comparison of the Garch with the realizations of Returns in sample and
% out of sample

Nrepl = length(ret_in);  % for direct comparison I simulate a series of 
                         % the same as in-sample series (also similar to out-of-sample) 

vol_sim = vol_lr;        % inital value of volatility
ret_sim(1) = sqrt(omega + alpha*0 + beta*vol_sim^2)*randn; 
                         % first simulated return: 1-step GARCH vol times N(0,1) 

for i = 2:Nrepl          % cycle for subsequent simulations
    vol_sim = sqrt(omega + alpha*ret_sim(i-1)^2 + beta*vol_sim^2);
                         % updated vol
    ret_sim(i) = vol_sim*randn;  
end

subplot(3,1,1)
plot(dates(1:t0-1),ret_in,'k')  
datetick('x','yyyy')
ylim([-0.1 0.1])
title('in-sample returns (03-06)')
subplot(3,1,2)
plot(dates(t0:end),ret_out,'k')  
datetick('x','yyyy')
ylim([-0.1 0.1])
title('out-of-sample returns (07-10)')
subplot(3,1,3)
plot(ret_sim,'k')  
xlim([1 Nrepl]), ylim([-0.1 0.1])
title('GARCH(1,1) simulated returns')


% 1-day forecast of the volatility using the GARCH

vol_GARCH(t0) = vol_lr;    % initial value of volatility for ret(t0)
                           % (''computed'' at date t0-1) 

for t = t0+1:T             
    vol_GARCH(t) = sqrt(omega + alpha*ret(t-1)^2 + beta*vol_GARCH(t-1)^2);
                       % GARCH updating of vol
                       % note: vol_GARCH(t) is the (conditional) vol of
                       % ret(t) computed at date t-1
end         

plot(dates(t0+1:end),vol_GARCH(t0+1:end),'k')
datetick('x','yyyy')
title('GARCH(1,1) volatility of Fiat ret on 07-10')


% 4 Estimation of the daily volatility:historical, EWMA volatility and
% comparison with Garh volatility
M = 250;       % length of the estimating window
for t = t0+1:T  
     vol_hist(t) = std(ret(t-M:t-1));   
               % historical vol of ret(t) computed at date t-1                    
end

vol_RM(t0) = std(ret_in);   % initial value of RiskMetrics vol
for t = t0+1:T
    vol_RM(t) = sqrt(0.94*vol_RM(t-1)^2 + 0.06*ret(t-1)^2);
               % RiskMetrics vol of ret(t) computed at date t-1
end

figure
plot(dates(t0+1:end),abs(ret(t0+1:end)),'k')
hold on
plot(dates(t0+1:end),vol_hist(t0+1:end),'b','linewidth',1.5)
plot(dates(t0+1:end),vol_RM(t0+1:end),'r','linewidth',1.5)
plot(dates(t0+1:end),vol_GARCH(t0+1:end),'m','linewidth',1.5)
datetick('x','yyyy')
title('Volatility estimates for FIAT on 07-10')
legend('|R_t|','hist vol','RiskMetrics vol','GARCH vol')


% VaR and ES at 1% daily horizon under normal assumption of returns 

pi_fiat = 100;

p = 0.01;    % order of VaR/ES (warning: 'alpha' is already in use)
const_VaR = -norminv(p);             % VaR of N(0,1)
const_ES = normpdf(norminv(p))/p;    % ES of N(0,1)

for t = t0+1:T
    V = pi_fiat*prices(t-1);      % portfolio value at date t-1
    VaR_hist(t) = V*const_VaR*vol_hist(t);   
               % estimate of VaR made at date t-1 for next date (t), using
               % historical vol
               % note: vol_hist(t) is used, which is the volatility of ret(t)
               % conditional at date t-1
    VaR_RM(t) = V*const_VaR*vol_RM(t);   % VaR with RiskMetrics vol
    VaR_GARCH(t) = V*const_VaR*vol_GARCH(t);  % VaR with GARCH vol
    
    ES_hist(t) = V*const_ES*vol_hist(t);   % same for ES
    ES_RM(t) = V*const_ES*vol_RM(t);
    ES_GARCH(t) = V*const_ES*vol_GARCH(t);
    
    PL(t) = V*ret(t);    % realized PL at date t: PL(t)=V(t)-V(t-1) 
end

figure
plot(dates(t0+1:end),VaR_hist(t0+1:end),'b')  
hold on
plot(dates(t0+1:end),VaR_RM(t0+1:end),'r')
plot(dates(t0+1:end),VaR_GARCH(t0+1:end),'m')
bar(dates(t0+1:end),-PL(t0+1:end),'k')   % -PL is plotted for direct comparison
datetick('x','yyyy')
title('VaR 1% estimates for FIAT portfolio on 07-10 vs. realized -PL (black)')
legend('hist vol VaR','RiskMetrics VaR','GARCH VaR')

figure
plot(dates(t0+1:end),ES_hist(t0+1:end),'b')  
hold on
plot(dates(t0+1:end),ES_RM(t0+1:end),'r')
plot(dates(t0+1:end),ES_GARCH(t0+1:end),'m')
bar(dates(t0+1:end),-PL(t0+1:end),'k')
datetick('x','yyyy')
title('ES 1% estimates for FIAT portfolio on 07-10 vs. realized -PL (black)')
legend('hist vol ES','RiskMetrics ES','GARCH ES')


% Var for a 10 days horizon

const = sqrt(10);     % scaling constant 
                      % (best to compute it outside the for cycle)
for t = t0+10:T       % start with the 10-day estimate for PL t0 -> t0+10
                      % computed at date t0
    VaR_hist10(t) = const*VaR_hist(t-9);  % VaR_hist(t-9) is the 1-day VaR for
                      % PL at day t-9, computed at day t-10
                      % it is multiplied by sqrt(10) in order to have the
                      % 10-day VaR for PL t-10 -> t 
    VaR_RM10(t) = const*VaR_RM(t-9);
    VaR_GARCH10(t) = const*VaR_GARCH(t-9);
    
    PL10(t) = sum(PL(t-9:t));    % PL t-10 -> t is sum of 10 intermediate daily PL
end

figure
plot(dates(t0+10:end),VaR_hist10(t0+10:end),'b')  
hold on
plot(dates(t0+10:end),VaR_RM10(t0+10:end),'r')
plot(dates(t0+10:end),VaR_GARCH10(t0+10:end),'m')
bar(dates(t0+10:end),-PL10(t0+10:end),'k')
datetick('x','yyyy')
title('10-day VaR 1% estimates for FIAT portfolio on 07-10 vs. realized 10-day -PL (black)')
legend('hist vol VaR','RiskMetrics VaR','GARCH VaR')


% Estimate the Garch(1,1) model with t-Student innovations and comparison
% with the result obtained with normal innovation

model = garch('ARCHLags',1,'GARCHLags',1,'Distribution','t');
          % here, 'model' is specified as (1,1) with t-Student innovations
          % note: the simpler syntax garch(1,1) may be used only for normal innovations 
model = estimate(model,ret_in);

omega_t = model.Constant;  
alpha_t = model.ARCH{1};  
beta_t = model.GARCH{1};  
vol_lr_t = sqrt(omega_t/(1-alpha_t-beta_t));

disp('omega,alpha,beta,vol_lr')
disp('normal innovations:')
disp([omega alpha beta vol_lr])
disp('t innovations:')
disp([omega_t alpha_t beta_t vol_lr_t])
      % note: all estimates are (slightly) different
      % ML estimators depend on the innovations distribution
 

% VaR and ES at 1% and daily horizon using t-Student assumption on returns
% and Garch volatility and comparison wiht results obtained under normal
% Garch

vol_GARCH_t(t0) = vol_lr_t;  

for t = t0+1:T
    vol_GARCH_t(t) = sqrt(omega_t + alpha_t*ret(t-1)^2 + beta_t*vol_GARCH_t(t-1)^2);
          % updating volatilities under t-student GARCH model just
          % estimated
end         

p = 0.01;
nu = model.Distribution.DoF;   % estimated parameter of the t-Student distribution
                               % for innovations (DoF = Degrees of Freedom)
                               % note: 'DoF' is a subfield of the field 'Distribution' 
q = tinv(p,nu);         % quantile of the t-Student with parameter nu
const_VaR = -sqrt((nu-2)/nu)*q; 
const_ES = sqrt((nu-2)/nu)*(nu+q^2)/(nu-1)*tpdf(q,nu)/p;   % see slides
                   % note: using 'tinv' and 'tpdf', nu must be
                   % provided too

for t = t0+1:T
    V = pi_fiat*prices(t-1);
    VaR_GARCH_t(t) = V*const_VaR*vol_GARCH_t(t);
    ES_GARCH_t(t) = V*const_ES*vol_GARCH_t(t);  
         % VaR and ES with t-Student GARCH volatilities
end

figure
plot(dates(t0+1:end),VaR_GARCH(t0+1:end),'k')  
hold on
plot(dates(t0+1:end),VaR_GARCH_t(t0+1:end),'r')
datetick('x','yyyy')
title('GARCH VaR 1% estimates for FIAT portfolio on 07-10')
legend('Normal innovations','t innovations')

figure
plot(dates(t0+1:end),ES_GARCH(t0+1:end),'k')  
hold on
plot(dates(t0+1:end),ES_GARCH_t(t0+1:end),'r')
datetick('x','yyyy')
title('GARCH ES 1% estimates for FIAT portfolio on 07-10')
legend('Normal innovations','t innovations')
