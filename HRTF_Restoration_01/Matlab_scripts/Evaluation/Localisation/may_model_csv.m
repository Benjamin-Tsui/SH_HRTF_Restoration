model = 'training_HRTF_08++_12_sparse';

% SADIE II hold outs
subject = '18';
hrtf_out = csvread(['SH_reconstruct/model_ouput/' model '_sub' subject '_out.csv']);
hrtf_in = csvread(['ML_HRTF_Data/Time_aligned/SH_HRTFs_1st_order_512_sparse_in_sub_' subject '_oct_3/SHed_hrtf_dB.txt']);
hrtf_ref = csvread(['ML_HRTF_Data/Time_aligned/SH_HRTFs_1st_order_512_sparse_in_sub_' subject '_oct_3/hrtf_dB.txt']);
angle_matched = csvread(['ML_HRTF_Data/Time_aligned/SH_HRTFs_1st_order_512_sparse_in_sub_' subject '_oct_3/angles.txt']);

% % Bernschutz data
% hrtf_out = csvread(['SH_reconstruct/model_ouput/' model '_bern_out.csv']);
% hrtf_in = csvread('ML_HRTF_Data/Time_aligned/SH_HRTFs_1st_order_512_sparse_in_bern_oct_3/SHed_hrtf_dB.txt');
% hrtf_ref = csvread('ML_HRTF_Data/Time_aligned/SH_HRTFs_1st_order_512_sparse_in_bern_oct_3/hrtf_dB.txt');
% angle_matched = csvread('ML_HRTF_Data/Time_aligned/SH_HRTFs_1st_order_512_sparse_in_bern_oct_3/angles.txt');


azi_idx = find(angle_matched(:,2) == 0 & angle_matched(:,1) >= -90 & angle_matched(:,1) <= 90);
fs = 48000;
target_azi = angle_matched(azi_idx, 1);
target_azi(target_azi > 180) = target_azi(target_azi > 180) - 360; % convert to -180 to 180
[target_azi, i] = sort(target_azi);
azi_idx = azi_idx(i);
out_azi = zeros(size(target_azi));

z_pad = zeros(1800,2);

% hrtf out to hrir out 
for n = 1:length(azi_idx)
    hrtf_azi = [hrtf_out(azi_idx(n), 1:256)', hrtf_out(azi_idx(n), 257:512)'];
    hrtf_azi = [hrtf_azi ; flipud(hrtf_azi)];
    %plot(hrtf);
    hrtf_azi =(10.^(hrtf_azi ./ 20));
    hrir_azi = ifft(hrtf_azi,512,1);
    hrir_azi = real(hrir_azi);
    hrir_azi = circshift(hrir_azi,256,1);
%     subplot(2,1,1);
%     plot(hrir_azi(:,1))
%     subplot(2,1,2);
%     plot(hrir_azi(:,2))
%     azi = angle_matched(azi_idx(n), 1);
%     title(num2str(azi))
    out = may2011([hrir_azi; z_pad],fs);
    out_azi(n) = may2011_estAzimuthGMM(out, 'HIST', 1, 0);
end
out_azi = out_azi .* -1; % convert direction to counter-clockwise

figure;
x_axis = linspace(-90, 90, length(target_azi));
plot(x_axis, target_azi)
hold on
plot(x_axis, out_azi)
title(['Model output prediction with Mays model'], 'Interpreter', 'none')
legend('target', 'may out','Location','northwest')
grid on
xlabel('target angle')
ylabel('responded angle')

mean_azi_error = nanmean(abs(out_azi - target_azi));
fprintf(['mean azimuth error (model output):', num2str(mean_azi_error) '\n'])

% hrtf out to hrir out 
for n = 1:length(azi_idx)
    hrtf_azi = [hrtf_in(azi_idx(n), 1:256)', hrtf_in(azi_idx(n), 257:512)'];
    hrtf_azi = [hrtf_azi ; flipud(hrtf_azi)];
    %plot(hrtf);
    hrtf_azi =(10.^(hrtf_azi ./ 20));
    hrir_azi = ifft(hrtf_azi,512,1);
    hrir_azi = real(hrir_azi);
    hrir_azi = circshift(hrir_azi,256,1);
%     subplot(2,1,1);
%     plot(hrir_azi(:,1))
%     subplot(2,1,2);
%     plot(hrir_azi(:,2))
%     azi = angle_matched(azi_idx(n), 1);
%     title(num2str(azi))
    out = may2011([hrir_azi; z_pad],fs);
    out_azi(n) = may2011_estAzimuthGMM(out, 'HIST', 1, 0);
end
out_azi = out_azi .* -1; % convert direction to counter-clockwise

figure;
x_axis = linspace(-90, 90, length(target_azi));
plot(x_axis, target_azi)
hold on
plot(x_axis, out_azi)
title(['SH input prediction with Mays model'], 'Interpreter', 'none')
legend('target', 'may out','Location','northwest')
grid on
xlabel('target angle')
ylabel('responded angle')

mean_azi_error = nanmean(abs(out_azi - target_azi));
fprintf(['mean azimuth error (SH input):', num2str(mean_azi_error) '\n'])

mean_sagittal_error(hrtf_out, hrtf_in, hrtf_ref, angle_matched);

% out = may2011(input,fs);