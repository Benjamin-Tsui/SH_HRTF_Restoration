function [diff, avgDiff_ERB, offset_F] = PspecComp_10(A, B, s, freq, limit, offset_F)

%{ 

freq = nfft
s = ones(length(A or B), 1)

VERSION 2.0

Function is part of a twin pair.

This twin compares the spectra of A and B in terms of PERCEPTUALLY WEIGHTED
error. For multi-dimensional inputs the comparason is made along the first
dimension. Averages are output for each column of data.

Perceptual error takes into account the lesser importance of quieter
sounds. Frequency bins are weighted with respect to the ISO 226 loudness
curves for an average listeneing level of 75 dB SPL. A comtribution half as
loud is deemed half as important.

The perceptual average difference is further weighted with respoect to ERB
bandwidth. Simply, this reduces the contibution to the average calculation
of higher frequency components where our ears are less sensitive. It's like
a logorythmic type average.

An itterative optimisation process is used to find the input normalisation
which results in the lowest error metric. This is generally somewhere
around the point at which the two input signals have the same mean value -
but can easily vary by a few dB. The optimum normalisation is different for
perceptual / absolute error metrics.

If you do not wish any normalisation to be applied, optional input offset_F
should be set to 'none'. A value of 0 will skip the optimal normalisation
process and just equate the mean values. Any other value will specify the
additional normalisation (in terms of dB gain) applied to B AFTER
normalisation with respect to the mean values of the two inputs.

*** A and B are expected to be input on a dB scale

%}

% Written by Calum Armstrong, Department of Electronics, The University 
% of York.
%
% Updates ---
% 05-03-2018: Release
% 05-03-2018: Normalisation patch (v1.1)
% 07-03-2018: Itterative Normalisation (v2.0)
% 07-03-2018: Initial Jump at start (v2.1)
% 07-03-2018: Removed p (listening / perceptual offset level) (v2.2)
% 08-03-2018: Split Absolute and Perceptual code
%             Refactored function (v2.3)
% 01-08-2018: Add solid angle weighiting

%% SETUP
% -----------------------------------------------------------------------

    % Check and format input vectors
    if size(A, 1) == 1
        A = A';
    end
    if size(B, 1) == 1
        B = B';
    end
    if size(freq, 1) == 1
        freq = freq';
    end

	% Test for input errors
    if size(A) ~= size(B)
        error('Input matricies A and B must be the same size');
    end
    
    % unless offset_F = 'none' normalise means
    if not( exist('offset_F', 'var')&&strcmp(offset_F, 'none') )
        % Normalize mean values of A and B
        avgA = mean(A(:)); avgB = mean(B(:));
        absNormOff = avgA - avgB;
        B = B + absNormOff;
    else
        offset_F = 0;
    end
    
    % Define what limits number of itterations
    if limit >= 1
        maxItterations = limit;
        minResolution = 0;
        lastItteration = limit;
    elseif limit > 0
        maxItterations = 1000;
        minResolution = limit;
    else
        error('Positive numeric value expected for input variable "limit"');    
    end
         
    % Set initial values for itterative normalization
    initInc = 0.2;
    scale = 0.4;
    offset = zeros(maxItterations, 1);
    avgPError = zeros(maxItterations, 1);

%% FUNCTION SETUP
% -----------------------------------------------------------------------
    
    % Useful variable for later
    sizeA = size(A);
    
    % ISO 226 Declarations
    iso226SPL = zeros(30, 91);
    iso226Freq = zeros(30, 91);
    Y = zeros(length(freq), 91);
    EL = zeros(length(freq), 91);

    % For all ISO standardised listening levels (ISO 226: 0-90dB SPL)
    for l = 1:91
        % Save equal loudness contour
        [iso226SPL(1:29, l), iso226Freq(1:29, l)] = iso226(l-1);
        iso226Freq(30, l) = 20000;
        iso226SPL(30, l) = iso226SPL(1, l);

        % Fit curve to equal loudness contour
        iso226Fit = fit(iso226Freq(:, l), iso226SPL(:, l), 'pchip');

        % Interpolate to input frequency bins and remove equivilant 1KHz
        % loudness offset
        Y(:, l) = iso226Fit(freq) - (l - 1);

        % Save the offset required in dB to equate the loudness of any
        % frequency bin to that of 1KHz for a given absolute loudness
        % (0-90dB SPL)
        % ... A.K.A. flip the equal loudness contour 1KHz offset!
        EL(:, l) = -Y(:, l) ;
    end
    
    % Calculate ERB bandwidths
    ERB = 0.108.*freq + 24.7;

    % Calculate ERB weights and repeat for input matrix multiplication
    ERBWeights_temp = (1./ERB) ./ max(1./ERB);
    ERBWeights = repmat(ERBWeights_temp, [1, sizeA(2:end)]);
    
%% ITTERATE TO FIND PERCEPTUALLY OPTIMUM INPUT NORMALISATION
% -----------------------------------------------------------------------
    
    if nargin < 6
        
        update = figure(); hold on;
        
        [~, temp, jump] = performComparason(A, B, 0);
        temp_weighted = temp .* s;
        avgPError(1) = sum(temp_weighted(:)) ./ sum(s(:));

        % Update figure
        figure(update);
        plot(0, avgPError(1), 'o'); hold on;
        drawnow
        
        % For each itteration
        for i = 2:maxItterations
            switch i
                case 2 % 1st (No offset)

                    offset(i) = jump;

                case 3 % 2nd (Try out positive offset)

                    offset(i) = jump + initInc;

                otherwise % All others

                    % If there was a reduction in perceptual error with
                    % respect to the previous offset, carry on increasing
                    % the offset in that direction (up or down)
                    if avgPError(i-1) < avgPError(i-2) 

                        offset(i) = offset(i-1) + initInc;

                    % If there was no improvement...
                    else 
                        % If this is only the third itteration, switch the
                        % offset direction, but keep the offset resolution
                        % the same
                        if i == 4

                            initInc = initInc * -1;
                            offset(i) = jump + initInc;

                        % Otherwise, carry on for two more samples then
                        % switch the offset direction and reduce the
                        % resultion (narrowing in on optimum value)
                        else

                            initInc = initInc * -scale;

                            % If the resolution has got too fine, quit
                            if abs(initInc) < minResolution
                                lastItteration = i-1;
                                break;
                            else
                                offset(i) = offset(i-1) + initInc;
                            end
                        end
                    end
            end

            % Perform comparason with calculated offset and save result
            [~, temp, ~] = performComparason(A, B, offset(i));
            temp_weighted = temp .* s;
            avgPError(i) = sum(temp_weighted(:)) ./ sum(s(:));

            % Update figure
            figure(update);
            plot(offset(i), avgPError(i), 'o'); hold on;
            drawnow

        end

        % Extract unique results
        [offset_U, ia, ~] = unique(offset(1:lastItteration));
        avgPError_U = avgPError(ia);

        % Find approximate minimum
        [~, minIndex] = min(avgPError_U);

        offset_F = offset_U(minIndex);
        fprintf('Optimum Normalisation Found: %d\n', offset_F);
    end

    
%% COMPARE SPECTRA WITH PERCEPTUALLY OPTIMAL NORMALISATION
% -----------------------------------------------------------------------
    
    % Perform comparason
    [diff, avgDiff_ERB, ~] = performComparason(A, B, offset_F);
    
    %close(update);
    
%% COMPARASON FUNCTION
% -----------------------------------------------------------------------
    function [pDiff, avgPDiff_ERB, jump] = performComparason(A, B, offset_F)  
    %% Re-Normalise inputs and amplify to 'Listening Level' amplitudes
    %  ... Average value set to 75dB SPL
    % ---------------------------------------------------------------------

        B = B + offset_F;

        % Normalize to 75dB
        meanValue = mean2([A B]);
        A = A + (75-meanValue);
        B = B + (75-meanValue);

% A = A + 20.5;
% B = B + 20.5;


    %% Account for Equal Loudness
    % ---------------------------------------------------------------------

        % Save matricies that select the corect loudness contour offsets to
        % use for each frequency bin for each input spectrum. This is the
        % integer rounded values of the input matricies with a maximum
        % value of 90 and a minimum value of 0.
        LC_A = min(round(A), 90);
        LC_A = max(LC_A, 0);
        LC_B = min(round(B), 90);
        LC_B = max(LC_B, 0);

        EL_A = EL(sub2ind(size(EL), repmat((1:sizeA(1))', [1, sizeA(2:end)]), LC_A+1));
        EL_B = EL(sub2ind(size(EL), repmat((1:sizeA(1))', [1, sizeA(2:end)]), LC_B+1));

        % Account for equal loudness / convert to phones scale
        A_EL = A + EL_A;
        B_EL = B + EL_B;


    %% Convert to Sones
    % ---------------------------------------------------------------------

        A_EL_Sones = 2.^((A_EL-40)/10);
        B_EL_Sones = 2.^((B_EL-40)/10);

    %% FIND LOUDNESS WEIGHTED DIFFERENCES
    % --------------------------------------------------------------------- 

        % Perceptual Difference
        pDiff = (B_EL_Sones - A_EL_Sones);

        avgPDiff_ERB = sum(ERBWeights .* abs(pDiff)) ./ sum(ERBWeights);

        jump = mean2(sum(ERBWeights .* (A_EL_Sones)) ./ sum(ERBWeights)) - mean2(sum(ERBWeights .* (B_EL_Sones)) ./ sum(ERBWeights));
        
    end
end
    
