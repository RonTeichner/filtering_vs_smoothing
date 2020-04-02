%% creating sm_vs_fl_different_f:
f = 0.01:0.01:0.99;
arg = f.^2 + sqrt(f.^4+1);
se = 0.5*arg;

denumenator = (0.5*arg+1).^2 - f.^2;
numenator = 0.5*(0.5*arg+1).*arg.^2;
ses = 0.5*(arg - numenator./denumenator);

se_db = 10*log10(se);
ses_db = 10*log10(ses);

min_se_db = min(se_db);
se_db = se_db - min_se_db;
ses_db = ses_db - min_se_db;

figure;
plot(f, se_db); hold all; plot(f, ses_db); xlabel('f'); ylabel('db'); 
legend('$\sigma_e^2$','$\sigma_{e,s}^2$','Interpreter','latex'); grid on; 
title('Estimation error variances for $\sigma_\omega^2=\sigma_v^2$','Interpreter','latex');
%title('$$Q \geq \frac{I_h H}{I_h H+I_z C}, b_1 \geq b_2$$','interpreter','latex')

%% creating delta_FS vs f:
f = 0.01:0.01:0.99;
gamma = f.^2 + sqrt(f.^4+1);
eta = 1;
numenator = 0.5*gamma + eta;
denumenator = (0.5*gamma + eta).^2 - f.^2*eta^2;
arg = gamma.*(numenator ./ denumenator);
deltaFS = arg ./ (2-0.5*arg);

figure; plot(f, deltaFS); xlabel('f'); grid on; title('$\Delta_{FS}$ for $\sigma_\omega^2=\sigma_v^2$','Interpreter','latex');
ylim([0, max(deltaFS)])