close all
% Load source image
img = imread('ResultD06_rgb.jpeg.jpeg');
% Load trained RCNN detector
load('detectorRCNN.mat');
% Detect objects in the source image
[bbox, scores, labels] = detect(detector, img);
% Filter out bounding boxes with scores less than 0.9
idx = scores > 0.9;
bbox = bbox(idx, :);
scores = scores(idx);
labels = labels(idx);
labels = cellstr(labels);
labels = replace(labels, 'Q', '');
% Create a table of the detected objects
detection_table = table(bbox, scores, labels);
% Calculate the centers of the bounding boxes
bbox_centers = [bbox(:,1) + bbox(:,3)/2, bbox(:,2) + bbox(:,4)/2];
% Convert from the [x y width height] bounding box format to the [xmin ymin
% xmax ymax] format for convenience.
%%%%%%%%%%%%%%%%%%%%%%%%% bounding boxes processing %%%%%%%%%%%%%%%%%%%%%%%%%%%
xmin = bbox(:, 1);
ymin = bbox(:, 2);
xmax = xmin + bbox(:, 3) - 1;
ymax = ymin + bbox(:, 4) - 1;
% Display the source image with the original bounding boxes
figure;
img = imcomplement(img);
imshow(img);
hold on;
for i = 1:size(bbox, 1)
    rectangle('Position', bbox(i,:), 'EdgeColor', 'b', 'LineWidth', 1);
    text(bbox(i,1), bbox(i,2)-10, labels{i}, 'Color', 'b', 'FontSize', 16);
end
hold off;
% Expand the bounding boxes by a small amount.
expansionAmount = 0.05;
xmin = (1-expansionAmount) * xmin;
% ymin = (1-expansionAmount) * ymin;
xmax = (1+expansionAmount) * xmax;
% ymax = (1+expansionAmount) * ymax;
% Calculate expanded bounding boxes [xmin ymin width height]
expandedBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];
% Display the source image with bounding box centers, x/y min/max lines
figure;
imshow(img);
hold on;
% scatter(bbox_centers(:, 1), bbox_centers(:, 2), 'r', 'filled');
% title('Bounding Box Centers');
% Draw x/y min/max lines for each bounding box
for i = 1:size(bbox, 1)
    % Draw x-min/max lines
    line([expandedBBoxes(i, 1) expandedBBoxes(i, 1)+expandedBBoxes(i, 3)-1 expandedBBoxes(i, 1)+expandedBBoxes(i, 3)-1 expandedBBoxes(i, 1) expandedBBoxes(i, 1)], ...
         [expandedBBoxes(i, 2) expandedBBoxes(i, 2) expandedBBoxes(i, 2)+expandedBBoxes(i, 4)-1 expandedBBoxes(i, 2)+expandedBBoxes(i, 4)-1 expandedBBoxes(i, 2)], ...
         'Color', 'g', 'LineWidth', 1);
    % Draw y-min/max lines
    line([expandedBBoxes(i, 1) expandedBBoxes(i, 1)+expandedBBoxes(i, 3)-1 expandedBBoxes(i, 1)+expandedBBoxes(i, 3)-1 expandedBBoxes(i, 1) expandedBBoxes(i, 1)], ...
         [expandedBBoxes(i, 2) expandedBBoxes(i, 2) expandedBBoxes(i, 2)+expandedBBoxes(i, 4)-1 expandedBBoxes(i, 2)+expandedBBoxes(i, 4)-1 expandedBBoxes(i, 2)], ...
         'Color', 'b', 'LineWidth', 1);
end
    hold off;
%%%%%%%%%%%%%%%%%%%% creation of graph %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
overlapRatio = bboxOverlapRatio(expandedBBoxes, expandedBBoxes);
% Set the overlap ratio between a bounding box and itself to zero to
% simplify the graph representation.
n = size(overlapRatio,1);
overlapRatio(1:n+1:n^2) = 0;
% Create the graph
g = graph(overlapRatio);
plot (g)
% Lisätään solmut graafiin ja asetetaan niihin nimet
numNodes = numel(labels);
nodeNames = cell(numNodes, 1);
for i = 1:numNodes
    nodeNames{i} = sprintf('%s [%d]', labels{i}, i);
end
g = graph(overlapRatio, nodeNames);
% Päivitetään graafin piirtokomento, jotta nimet näkyvät solmuissa
h = plot(g, 'NodeLabel', g.Nodes.Name);
% Sijoitetaan solmujen nimet paremmin näkyviin
layout(h, 'force');
% Näytetään graafi
title('Overlap Graph with Labels');
%%%%%%%%%%%%%%%%%%% connected text reg
%%%%%%%%%%%%%%%%%%% connected text regions finding within the graph %%%%%%%
% Find the connected text regions within the graph
componentIndices = conncomp(g);
% Conncomp on jo suoritettu ja componentIndices on saatavilla
numComponents = max(componentIndices);
componentBBoxes = zeros(numComponents, 4); % [xmin, ymin, xmax, ymax]
% Laske jokaisen komponentin yhteinen alue
for i = 1:numComponents
    % Hae komponentin bounding boxit
    componentBoxes = bbox(componentIndices == i, :);
    % Lasketaan yhteinen alue
    xmin = min(componentBoxes(:,1));
    ymin = min(componentBoxes(:,2));
    xmax = max(componentBoxes(:,1) + componentBoxes(:,3));
    ymax = max(componentBoxes(:,2) + componentBoxes(:,4));
    componentBBoxes(i, :) = [xmin, ymin, xmax, ymax];
end
% Laske keskipisteet ja järjestä komponentit
centers = [(componentBBoxes(:,1) + componentBBoxes(:,3)) / 2, ...
           (componentBBoxes(:,2) + componentBBoxes(:,4)) / 2];
[~, sortedIndices] = sortrows(centers, [2, 1]); % Ensin Y, sitten X
% % Tulosta tai tallenna järjestettyjen komponenttien tiedot
% for i = 1:length(sortedIndices)
%     compIdx = sortedIndices(i);
%     fprintf('Komponentti %d: xmin=%f, ymin=%f, xmax=%f, ymax=%f\n', ...
%             compIdx, componentBBoxes(compIdx, 1), componentBBoxes(compIdx, 2), ...
%             componentBBoxes(compIdx, 3), componentBBoxes(compIdx, 4));
% end
% Käydään läpi jokainen komponentti
for i = 1:length(sortedIndices)
    compIdx = sortedIndices(i);
    fprintf('Komponentti %d:\n', compIdx);

    % Etsi tämän komponentin bounding boxit ja labelit
    componentBoxIndices = find(componentIndices == compIdx);
    componentBoxes = bbox(componentBoxIndices, :);
    componentLabels = labels(componentBoxIndices);

    % Järjestä komponentin tekstit x-koordinaattien perusteella
    [~, componentOrder] = sort(componentBoxes(:, 1));
    sortedComponentLabels = componentLabels(componentOrder);

    % Tulosta järjestetyt labelit
    for j = 1:length(sortedComponentLabels)
        fprintf('Label: %s\n', sortedComponentLabels{j});
    end
    fprintf('\n');
end
