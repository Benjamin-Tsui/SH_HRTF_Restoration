ITA_HRTF = SOFAload('MRT10.sofa'); % ITA
max_ITA = max(max(max(ITA_HRTF.Data.IR)));
[max_ITA_loc1, ~, ~] = find(ITA_HRTF.Data.IR == max_ITA);
ITA_hrtf_left = squeeze(ITA_HRTF.Data.IR(max_ITA_loc1, 1, :));
ITA_hrtf_right = squeeze(ITA_HRTF.Data.IR(max_ITA_loc1, 2, :));
figure
subplot(3, 2, 1)
plot(ITA_hrtf_left)
hold on
plot(ITA_hrtf_right)
title('ITA maximum amplutude HRTF')
xlabel('time')
ylabel('amplitude')

ARI_HRTF = SOFAload('hrtf b_nh110.sofa'); % ARI
max_ARI = max(max(max(ARI_HRTF.Data.IR)));
[max_ARI_loc1, ~, ~] = find(ARI_HRTF.Data.IR == max_ARI);
ARI_hrtf_left = squeeze(ARI_HRTF.Data.IR(max_ARI_loc1, 1, :));
ARI_hrtf_right = squeeze(ARI_HRTF.Data.IR(max_ARI_loc1, 2, :));
subplot(3, 2, 2)
plot(ARI_hrtf_left)
hold on
plot(ARI_hrtf_right)
title('ARI maximum amplutude HRTF')
xlabel('time')
ylabel('amplitude')

CIPIC_HRTF = SOFAload('subject_011.sofa'); % CIPIC
max_CIPIC = max(max(max(CIPIC_HRTF.Data.IR)));
[max_CIPIC_loc1, ~, ~] = find(CIPIC_HRTF.Data.IR == max_CIPIC);
CIPIC_hrtf_left = squeeze(CIPIC_HRTF.Data.IR(max_CIPIC_loc1, 1, :));
CIPIC_hrtf_right = squeeze(CIPIC_HRTF.Data.IR(max_CIPIC_loc1, 2, :));
subplot(3, 2, 3)
plot(CIPIC_hrtf_left)
hold on
plot(CIPIC_hrtf_right)
title('CIPIC maximum amplutude HRTF')
xlabel('time')
ylabel('amplitude')

SADIE_HRTF = SOFAload('SADIE_015_DFC_256_order_fir_48000.sofa'); % SADIE
max_SADIE = max(max(max(SADIE_HRTF.Data.IR)));
[max_SADIE_loc1, ~, ~] = find(SADIE_HRTF.Data.IR == max_SADIE);
SADIE_hrtf_left = squeeze(SADIE_HRTF.Data.IR(max_SADIE_loc1, 1, :));
SADIE_hrtf_right = squeeze(SADIE_HRTF.Data.IR(max_SADIE_loc1, 2, :));
subplot(3, 2, 4)
plot(SADIE_hrtf_left)
hold on
plot(SADIE_hrtf_right)
title('SADIE maximum amplutude HRTF')
xlabel('time')
ylabel('amplitude')

IRCAM_HRTF = SOFAload('irc_1033.sofa'); % IRCAM
max_IRCAM = max(max(max(IRCAM_HRTF.Data.IR)));
[max_IRCAM_loc1, ~, ~] = find(IRCAM_HRTF.Data.IR == max_IRCAM);
IRCAM_hrtf_left = squeeze(IRCAM_HRTF.Data.IR(max_IRCAM_loc1, 1, :));
IRCAM_hrtf_right = squeeze(IRCAM_HRTF.Data.IR(max_IRCAM_loc1, 2, :));
subplot(3, 2, 5)
plot(IRCAM_hrtf_left)
hold on
plot(IRCAM_hrtf_right)
title('IRCAM maximum amplutude HRTF')
xlabel('time')
ylabel('amplitude')

RIEC_HRTF = SOFAload('RIEC_hrir_subject_039.sofa'); % RIEC
max_RIEC = max(max(max(RIEC_HRTF.Data.IR)));
[max_RIEC_loc1, ~, ~] = find(RIEC_HRTF.Data.IR == max_RIEC);
RIEC_hrtf_left = squeeze(RIEC_HRTF.Data.IR(max_RIEC_loc1, 1, :));
RIEC_hrtf_right = squeeze(RIEC_HRTF.Data.IR(max_RIEC_loc1, 2, :));
subplot(3, 2, 6)
plot(RIEC_hrtf_left)
hold on
plot(RIEC_hrtf_right)
title('RIEC maximum amplutude HRTF')
xlabel('time')
ylabel('amplitude')


