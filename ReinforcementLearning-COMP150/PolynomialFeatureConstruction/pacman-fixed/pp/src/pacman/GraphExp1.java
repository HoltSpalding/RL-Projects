package pacman;


import java.awt.BasicStroke;
import java.awt.Color;
import java.util.ArrayList;

import javax.swing.JPanel;
import java.util.ArrayDeque;
import java.util.ArrayList;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.DeviationRenderer;
import org.jfree.data.time.RegularTimePeriod;
import org.jfree.data.time.Week;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.YIntervalSeries;
import org.jfree.data.xy.YIntervalSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RectangleInsets;
import org.jfree.ui.RefineryUtilities;


public class GraphExp1 extends ApplicationFrame {

    
    public GraphExp1(String title, ArrayList<Double> SarsaXs, ArrayList<Double> QXs, int numcurves) {
        super(title);
        JFreeChart chart = createChart(createDataset(SarsaXs, QXs, numcurves));
        JPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));
        setContentPane(chartPanel);
    }

    /**
     * Creates a sample dataset.
     *
     * @return a sample dataset.
     */
    private static XYDataset createDataset(ArrayList<Double> SarsaXs, ArrayList<Double> QXs, int numcurves) {

        YIntervalSeries series1 = new YIntervalSeries("Sarsa");
        YIntervalSeries series2 = new YIntervalSeries("Q-Learning");
        int cycles = SarsaXs.size()/numcurves;
/*        int sarsize = SarsaXs.size();
        int qsize = QXs.size();*/
        for (int i = 0; i < cycles; i++) {
        	double stdev = 0;
        	double acc = 0;
        	for (int j = 0; j < numcurves; j++) {
        		int idx = i+j*cycles;
        		acc += SarsaXs.get(idx);
        	}
        	double mean = acc / numcurves;
        	for (int j = 0; j < numcurves; j++) {
        		double num = SarsaXs.get(i+j*cycles);
        		stdev += (num - mean) * (num - mean);	
        	}
        	stdev = Math.sqrt(stdev/numcurves);
        	System.out.println(i);
        	series1.add(i, mean, mean - stdev, mean + stdev);


        }
        for (int i = 0; i < cycles; i++) {
        	double stdev = 0;
        	double acc = 0;
        	for (int j = 0; j < numcurves; j++) {
        		acc += QXs.get(i+j*cycles);
        	}
        	double mean = acc / numcurves;
        	for (int j = 0; j < numcurves; j++) {
        		double num = QXs.get(i+j*cycles);
        		stdev += (num - mean) * (num - mean);	
        	}
        	stdev = Math.sqrt(stdev/numcurves);
        	series2.add(i, mean, mean - stdev, mean + stdev);
        }
      
        

        YIntervalSeriesCollection dataset = new YIntervalSeriesCollection();
        dataset.addSeries(series1);
        dataset.addSeries(series2);

        return dataset;

    }


    private static JFreeChart createChart(XYDataset dataset) {

        // create the chart...
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Sarsa vs QLearning Pacman Performance With Poly Features", 
                "Training Cycles", 
                "Pacman Score", 
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
            );



        // get a reference to the plot for further customisation...
        XYPlot plot = (XYPlot) chart.getPlot();
        plot.setDomainPannable(false);
        plot.setRangePannable(true);
        plot.setInsets(new RectangleInsets(5, 5, 5, 20));

        DeviationRenderer renderer = new DeviationRenderer(true, false);
        renderer.setSeriesStroke(0, new BasicStroke(3.0f, BasicStroke.CAP_ROUND,
                BasicStroke.JOIN_ROUND));
        renderer.setSeriesStroke(0, new BasicStroke(3.0f,
                BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        renderer.setSeriesStroke(1, new BasicStroke(3.0f,
                BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        renderer.setSeriesFillPaint(0, new Color(255, 200, 200));
        renderer.setSeriesFillPaint(1, new Color(200, 200, 255));
        plot.setRenderer(renderer);

        // change the auto tick unit selection to integer units only...
        NumberAxis yAxis = (NumberAxis) plot.getRangeAxis();
        yAxis.setAutoRangeIncludesZero(false);
        yAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());

        return chart;

    }


public static void main(ArrayList<Double> SarsaXs, ArrayList<Double> QXs, int numcurves) {
    	System.out.println("helllo");
        GraphExp1 demo = new GraphExp1(
                "Sarsa vs QLearning Pacman Performance With Poly Features", SarsaXs, QXs, numcurves);
        demo.pack();
        RefineryUtilities.centerFrameOnScreen(demo);
        demo.setVisible(true);

    }

}