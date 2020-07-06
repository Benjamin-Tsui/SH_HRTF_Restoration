function [model, subject] = plot_output_hrtfs_SADIE(azi, ele, subject, model)
%PLOT_OUTPUT_HRTFS_SUB18 Summary of this function goes here
%   Detailed explanation goes here

% example:
% plot_output_hrtfs_SADIE(90, 0, '18', 'training_HRTF_08++_03_sparse');
% subject = '18' or  '19' or  '20';
% model = 'training_HRTF_08++_03_sparse';

% note: output for debugging

%% csv results 
% SADIE II hold outs
% subject = '19';
    
hrtf_out = csvread(['SH_reconstruct/model_ouput/' model '_sub' subject '_out.csv']);
hrtf_in = csvread(['ML_HRTF_Data/Time_aligned/SH_HRTFs_1st_order_512_sparse_in_sub_' subject '_oct_3/SHed_hrtf_dB.txt']);
hrtf_tar = csvread(['ML_HRTF_Data/Time_aligned/SH_HRTFs_1st_order_512_sparse_in_sub_' subject '_oct_3/hrtf_dB.txt']);
angle_matched = csvread(['ML_HRTF_Data/Time_aligned/SH_HRTFs_1st_order_512_sparse_in_sub_' subject '_oct_3/angles.txt']);

% hrtf_out = add_back_low_freq(hrtf_out, hrtf_in);

hrtf_dB_left_in = hrtf_in(:, 1:256);
hrtf_dB_right_in = hrtf_in(:, 257:512);
hrtf_dB_left_out = hrtf_out(:, 1:256);
hrtf_dB_right_out = hrtf_out(:, 257:512);
hrtf_dB_left_tar = hrtf_tar(:, 1:256);
hrtf_dB_right_tar = hrtf_tar(:, 257:512);

fs = 48000;
nfft = 512;
%% plot angle

[~,i] = min(sum(abs(angle_matched(:,1:2)- [azi ele]),2));

x_value = linspace(1, 24000, 256);

figure
semilogx(x_value,hrtf_dB_left_out(i, :),'Color',[0.9600 0.5250 0.0080],'LineWidth', 1.6)
hold on
semilogx(x_value,hrtf_dB_left_tar(i, :),'Color',[0.2660 0.6820 0.0880],'LineWidth', 1.6)
hold on
semilogx(x_value,hrtf_dB_left_in(i, :),'Color',[0 0.5570 0.8110],'LineWidth', 1.6)
xlim([20 28000])
% xline(700,'--r','\fontsize{20}700 Hz','LineWidth',3,'LabelVerticalAlignment','bottom',...
%     'LabelOrientation','horizontal');
grid on
xlabel('Hz','FontSize',18)
ylabel('dB','FontSize',18)
legend('model out (reconstructed)','target (actual measurement)','input (1st order SH)', 'location', 'southwest','FontSize',18, 'FontWeight', 'Bold' )
th1 = title (['\fontsize{21} SADIE subject ' subject ' HRTF reconsturucted result (left channel, azi: ' num2str(azi) ', ele: ' num2str(ele) ')']);

% legend('actual measurement','1st order SH interpolated', 'location', 'southwest','FontSize',18, 'FontWeight', 'Bold' )
% th1 = title (['\fontsize{19} SADIE subject ' subject ' interpolated HRTF vs actual meausrement (left channel, azi: ' num2str(azi) ', ele: ' num2str(ele) ')']);

% get the position of the title
titlePos1 = get( th1 , 'position');
% change the x value  to 1.5
titlePos1(2) = 11.1;
% titlePos1(2) = 11.2;
% update the position
set( th1 , 'position' , titlePos1);

set(gcf, 'Position',  [100, 200, 1000, 600])
set(gcf,'color','w');

figure
semilogx(x_value,hrtf_dB_right_out(i, :),'Color',[0.9600 0.5250 0.0080],'LineWidth', 1.6)
hold on
semilogx(x_value,hrtf_dB_right_tar(i, :),'Color',[0.2660 0.6820 0.0880],'LineWidth', 1.6)
hold on
semilogx(x_value,hrtf_dB_right_in(i, :),'Color',[0 0.5570 0.8110],'LineWidth', 1.6)
% xline(700,'--r','\fontsize{20}700 Hz','LineWidth',3,'LabelVerticalAlignment','bottom',...
%     'LabelOrientation','horizontal');
xlim([20 28000])
grid on
xlabel('Hz','FontSize',18)
ylabel('dB','FontSize',18)
legend('model out (reconstructed)','target (actual measurement)','input (1st order SH)', 'location', 'southwest','FontSize',18, 'FontWeight', 'Bold' )
th2 = title (['\fontsize{21} SADIE subject ' subject ' HRTF reconsturucted result (right channel, azi: ' num2str(azi) ', ele: ' num2str(ele) ')']);

% legend('actual measurement','1st order SH interpolated', 'location', 'southwest','FontSize',18, 'FontWeight', 'Bold' )
% th2 = title (['\fontsize{19} SADIE subject ' subject ' interpolated HRTF vs actual meausrement (right channel, azi: ' num2str(azi) ', ele: ' num2str(ele) ')']);

% get the position of the title
titlePos2 = get( th2 , 'position');
% change the x value  to 1.5
titlePos2(2) = 1.5;
% update the position
set( th2 , 'position' , titlePos2);

set(gcf, 'Position',  [200, 100, 1000, 600])
set(gcf,'color','w');


end

