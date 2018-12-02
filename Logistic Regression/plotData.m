function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
figure; 

        neg = find(y == 0);
        pos = find(y==1); 
        
% Plot Examples 
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7); 
hold on;
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y','MarkerSize', 7);

hold off;

end
