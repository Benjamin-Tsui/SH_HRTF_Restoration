function [shuffled_feature, shuffled_label] = shufflerow(feature, label, random_seed)
if nargin == 2
    random_seed = 'shuffle';
end
rng(random_seed);
r = size(feature, 1);
shuffledRow = randperm(r);
shuffled_feature = feature(shuffledRow, :);
shuffled_label = label(shuffledRow, :);
end