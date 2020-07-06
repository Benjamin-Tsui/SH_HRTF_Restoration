% %% csv results 
% % SADIE II hold outs
% subject = '19';
% hrtf_out = csvread(['SH_reconstruct/model_ouput/training_HRTF_08++_03_sparse_sub' subject '_out.csv']);
% hrtf_in = csvread(['ML_HRTF_Data/Time_aligned/SH_HRTFs_1st_order_512_sparse_in_sub_' subject '_oct_3/SHed_hrtf_dB.txt']);
% hrtf_tar = csvread(['ML_HRTF_Data/Time_aligned/SH_HRTFs_1st_order_512_sparse_in_sub_' subject '_oct_3/hrtf_dB.txt']);
% angle_matched = csvread(['ML_HRTF_Data/Time_aligned/SH_HRTFs_1st_order_512_sparse_in_sub_' subject '_oct_3/angles.txt']);
% 
% % hrtf_out = add_back_low_freq(hrtf_out, hrtf_in);
% 
% hrtf_dB_left_in = hrtf_in(:, 1:256);
% hrtf_dB_right_in = hrtf_in(:, 257:512);
% hrtf_dB_left_out = hrtf_out(:, 1:256);
% hrtf_dB_right_out = hrtf_out(:, 257:512);
% hrtf_dB_left_tar = hrtf_tar(:, 1:256);
% hrtf_dB_right_tar = hrtf_tar(:, 257:512);
% 
% fs = 48000;
% nfft = 512;

%% plot angle
azi = 0;
ele = 0;

[~,i] = min(sum(abs(angle_matched(:,1:2)- [azi ele]),2));

x_value = linspace(1, 24000, 256);

figure
semilogx(x_value,hrtf_dB_left_out(i, :),'Color',[0.9600 0.5250 0.0080],'LineWidth', 1.6)
hold on
semilogx(x_value,hrtf_dB_left_tar(i, :),'Color',[0.2660 0.6820 0.0880],'LineWidth', 1.6)
semilogx(x_value,hrtf_dB_left_in(i, :),'Color',[0 0.5570 0.8110],'LineWidth', 1.6)
xlim([20 28000])
grid on
xlabel('Hz','FontSize',12)
ylabel('dB','FontSize',12)
legend('model out (reconstructed)','target (actual measurement)','input (1st order SH)', 'location', 'southwest','FontSize',12, 'FontWeight', 'Bold' )
title (['\fontsize{16}HRTF reconsturucted result (left channel, azi: ' num2str(azi) ', ele: ' num2str(ele) ')'])
set(gcf, 'Position',  [100, 200, 1000, 600])
set(gcf,'color','w');

figure
semilogx(x_value,hrtf_dB_right_out(i, :),'Color',[0.9600 0.5250 0.0080],'LineWidth', 1.6)
hold on
semilogx(x_value,hrtf_dB_right_tar(i, :),'Color',[0.2660 0.6820 0.0880],'LineWidth', 1.6)
semilogx(x_value,hrtf_dB_right_in(i, :),'Color',[0 0.5570 0.8110],'LineWidth', 1.6)
xlim([20 28000])
grid on
xlabel('Hz','FontSize',12)
ylabel('dB','FontSize',12)
legend('model out (reconstructed)','target (actual measurement)','input (1st order SH)', 'location', 'southwest','FontSize',12, 'FontWeight', 'Bold' )
title (['\fontsize{16}HRTF reconsturucted result (right channel, azi: ' num2str(azi) ', ele: ' num2str(ele) ')'])
set(gcf, 'Position',  [200, 100, 1000, 600])
set(gcf,'color','w');

