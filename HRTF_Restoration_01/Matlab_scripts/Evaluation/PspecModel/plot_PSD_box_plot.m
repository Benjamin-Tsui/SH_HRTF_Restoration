function  plot_PSD_box_plot(psdResultsAMB,psdResultsAIO)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

dist = 0.15;

psdResultsAMBX = (1:length(psdResultsAMB(1,:)))-dist;
psdResultsAIOX = (1:length(psdResultsAMB(1,:)))+dist;

combined = [psdResultsAMB psdResultsAIO];
combinedX = [psdResultsAMBX psdResultsAIOX];

[combinedX, sortOrder] = sort(combinedX,'ascend');
combined = combined(sortOrder);

    plotColour = get(gca,'colororder');



% h  = figure('Color', [1 1 1]); 
% p1 = plot(psdResultsAMBX,psdResultsAMB,'s','Color',plotColour(1,:));
% set(p1, 'MarkerSize', 8,'LineWidth',1.3,'MarkerFaceColor','w')
% 
% hold on
% p2 = plot(psdResultsAIOX,psdResultsAIO,'s','Color',plotColour(2,:));
% set(p2, 'MarkerSize', 8,'LineWidth',1.3,'MarkerFaceColor','w')


% for i = 1:(length(psdResultsAMB(1,:))-1)
% line([i+0.5 i+0.5],[0 3],'Color',[0.8 0.8 0.8],'LineStyle',':','LineWidth',0.2);
% end
% 
% % p2 = plot(psdResultsAIOX,psdResultsAIO,'b^');
% % set(p2, 'MarkerSize', 9,'LineWidth',1.3)
% 
% set(gca,'fontsize', 14);
% % e1 = plot(combinedX,combined,'k'); % to put a line in between each result
% 
% 
% 
% for i = 1:(length(psdResultsAMB(1,:))-1)
% line([i+0.5 i+0.5],[-5 105],'Color',[0.8 0.8 0.8],'LineStyle',':','LineWidth',0.2);
% end



er1 = boxplot(psdResultsAMB, 'Positions', psdResultsAMBX, 'Widths', 0.15,'Colors',plotColour(1,:)); 
hold on
er2 = boxplot(psdResultsAIO, 'Positions', psdResultsAIOX, 'Widths', 0.15,'Colors',plotColour(2,:)); 
lineWidth = 1.5; lineCover=3*lineWidth;
a = [findall(gcf,'Marker','none') findall(gcf,'Marker','.')];
set(a,'LineWidth',lineWidth,'Marker','.','MarkerSize',lineCover);
% er1.Color = [0.3 0.3 0.3];
% er2.Color = [0.3 0.3 0.3];
% er1.LineWidth = 1.1;
% er2.LineWidth = 1.1;

p1 = plot(psdResultsAMBX,nanmedian(psdResultsAMB),'s','Color',plotColour(1,:));
set(p1, 'MarkerSize', 10,'LineWidth',1.6,'MarkerFaceColor',plotColour(1,:))
p2 = plot(psdResultsAIOX,nanmedian(psdResultsAIO),'s','Color',plotColour(2,:));
set(p2, 'MarkerSize', 10,'LineWidth',1.6,'MarkerFaceColor',plotColour(2,:))

set(gca,'fontsize', 18);
set(gcf,'color','w');
ylabel('PSD Score');
xlabel('HRTF dataset');
set(gca,'YGrid','on') 
set(gca,'YMinorGrid','on') 
xticks([1 2 3 4 5 6 7 8 9 10]);
set(gca,'xticklabels',{'SADIE 18 (training data)','SADIE 19 (test data)','SADIE 20 (test data)', 'Bernschutz (test data)'})
legend({'SH input', 'Model output'}, 'location', 'northeast');

% xlim([0.5 (length(psdResultsAMB)+0.5)])
ylim([0 6.5])
% pbaspect([1 1 1])

%  pbaspect([0.75 1 1])


% set(gca, 'YDir', 'reverse')


hold off

end

