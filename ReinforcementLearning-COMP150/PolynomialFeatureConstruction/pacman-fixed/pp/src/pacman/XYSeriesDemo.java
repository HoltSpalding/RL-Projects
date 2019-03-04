package pacman;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;
import java.awt.Color; 
import java.awt.BasicStroke; 

import org.jfree.chart.ChartPanel; 
import org.jfree.chart.JFreeChart; 
import org.jfree.data.xy.XYDataset; 
import org.jfree.data.xy.XYSeries; 
import org.jfree.ui.ApplicationFrame; 
import org.jfree.ui.RefineryUtilities; 
import org.jfree.chart.plot.XYPlot; 
import org.jfree.chart.ChartFactory; 
import org.jfree.chart.plot.PlotOrientation; 
import org.jfree.data.xy.XYSeriesCollection; 
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;

public class XYSeriesDemo extends ApplicationFrame {

/**
 * A demonstration application showing an XY series containing a null value.
 *
 * @param title  the frame title.
 */
public XYSeriesDemo(final String title, ArrayList<Double> Graph1Xs, ArrayList<Double> Graph2Xs) {

    super(title);
    final XYSeries series = new XYSeries("Graph1");
    for(int i = 0; i < Graph1Xs.size(); i++) {
    	series.add(i, Graph1Xs.get(i));
    	System.out.println(Graph1Xs.get(i));
    }
    final XYSeries series2 = new XYSeries("Graph2");
    for(int i = 0; i < Graph2Xs.size(); i++) {
    	series2.add(i, Graph2Xs.get(i));
    }
    final XYSeriesCollection data = new XYSeriesCollection( series );
    data.addSeries(    series2 );

    final JFreeChart chart = ChartFactory.createXYLineChart(
        "Depth Features vs Custom Features Effect On Sarsa Pacman Performance",
        "Training Cycles", 
        "Pacman Score", 
        data,
        PlotOrientation.VERTICAL,
        true,
        true,
        false
    );

    final ChartPanel chartPanel = new ChartPanel(chart);
    chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));
    final XYPlot plot = chart.getXYPlot( ); 
    XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer( );
    renderer.setSeriesPaint( 0 , Color.RED );
    renderer.setSeriesPaint( 1 , Color.GREEN );
    renderer.setSeriesStroke( 0 , new BasicStroke( 4.0f ) );
    renderer.setSeriesStroke( 1 , new BasicStroke( 3.0f ) );
    plot.setRenderer( renderer ); 
    setContentPane( chartPanel ); 
    setContentPane(chartPanel);

}

/*public static void m(ArrayList<Double> x1, ArrayList<Double> x2) {

    final XYSeriesDemo demo = new XYSeriesDemo("XY Series Demo");
    demo.pack();
    RefineryUtilities.centerFrameOnScreen(demo);
    demo.setVisible(true);

}*/
public static void m(ArrayList<Double> Graph1Xs, ArrayList<Double> Graph2Xs) {

    final XYSeriesDemo demo = new XYSeriesDemo("Depth Features vs Custom Features Effect On Sarsa Pacman Performance",Graph1Xs, Graph2Xs);
    demo.pack();
    RefineryUtilities.centerFrameOnScreen(demo);
    demo.setVisible(true);

}

}
