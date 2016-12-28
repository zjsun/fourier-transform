package it.polito.teaching.cv;

import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.io.ByteArrayInputStream;
import java.io.File;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;

/**
 * The controller associated to the only view of our application. The
 * application logic is implemented here. It handles the button for opening an
 * image and perform all the operation related to the Fourier transformation and
 * antitransformation.
 *
 * @author <a href="mailto:luigi.derussis@polito.it">Luigi De Russis</a>
 * @version 1.1 (2015-11-03)
 * @since 1.0 (2013-12-11)
 */
public class FourierController {
    // images to show in the view
    @FXML
    private ImageView originalImage;
    @FXML
    private ImageView transformedImage;
    @FXML
    private ImageView antitransformedImage;
    // a FXML button for performing the transformation
    @FXML
    private Button transformButton;
    // a FXML button for performing the antitransformation
    @FXML
    private Button antitransformButton;

    // the main stage
    private Stage stage;
    // the JavaFX file chooser
    private FileChooser fileChooser;
    // support variables
    private Mat image;
    private MatVector planes;
    // the final complex image
    private Mat complexImage;

    /**
     * Init the needed variables
     */
    protected void init() {
        this.fileChooser = new FileChooser();
        this.image = new Mat();
        this.planes = new MatVector();
        this.complexImage = new Mat();
    }

    /**
     * Load an image from disk
     */
    @FXML
    protected void loadImage() {
        // show the open dialog window
        File file = this.fileChooser.showOpenDialog(this.stage);
        if (file != null) {
            // read the image in gray scale
            this.image = imread(file.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            // show the image
            this.originalImage.setImage(mat2Image(this.image));
            // set a fixed width
            this.originalImage.setFitWidth(250);
            // preserve image ratio
            this.originalImage.setPreserveRatio(true);
            // update the UI
            this.transformButton.setDisable(false);

            // empty the image planes and the image views if it is not the first
            // loaded image
            if (this.planes.size() > 0) {
                this.planes.deallocate();
                this.planes = new MatVector();
                this.transformedImage.setImage(null);
                this.antitransformedImage.setImage(null);
            }

        }
    }

    /**
     * The action triggered by pushing the button for apply the dft to the
     * loaded image
     */
    @FXML
    protected void transformImage() {
        // optimize the dimension of the loaded image
        Mat padded = this.optimizeImageDim(this.image);
        padded.convertTo(padded, CV_32F);
        // prepare the image planes to obtain the complex image
//        this.planes.put(padded);
//        this.planes.put(Mat.zeros(padded.size(), CV_32F));
        this.planes.put(padded, Mat.zeros(padded.size(), CV_32F).asMat());

        // prepare a complex image for performing the dft
        merge(this.planes, this.complexImage);

        // dft
        dft(this.complexImage, this.complexImage);

        // optimize the image resulting from the dft operation
        Mat magnitude = this.createOptimizedMagnitude(this.complexImage);

        // show the result of the transformation as an image
        this.transformedImage.setImage(mat2Image(magnitude));
        // set a fixed width
        this.transformedImage.setFitWidth(250);
        // preserve image ratio
        this.transformedImage.setPreserveRatio(true);

        // enable the button for performing the antitransformation
        this.antitransformButton.setDisable(false);
        // disable the button for applying the dft
        this.transformButton.setDisable(true);
    }

    /**
     * The action triggered by pushing the button for apply the inverse dft to
     * the loaded image
     */
    @FXML
    protected void antitransformImage() {
        idft(this.complexImage, this.complexImage);

        Mat restoredImage = new Mat();
        split(this.complexImage, this.planes);
        normalize(this.planes.get(0), restoredImage, 0, 255, NORM_MINMAX, CV_8UC1, Mat.EMPTY);

        this.antitransformedImage.setImage(mat2Image(restoredImage));
        // set a fixed width
        this.antitransformedImage.setFitWidth(250);
        // preserve image ratio
        this.antitransformedImage.setPreserveRatio(true);

        // disable the button for performing the antitransformation
        this.antitransformButton.setDisable(true);
    }

    /**
     * Optimize the image dimensions
     *
     * @param image the {@link Mat} to optimize
     * @return the image whose dimensions have been optimized
     */
    private Mat optimizeImageDim(Mat image) {
        // init
        Mat padded = new Mat();
        // get the optimal rows size for dft
        int addPixelRows = getOptimalDFTSize(image.rows());
        // get the optimal cols size for dft
        int addPixelCols = getOptimalDFTSize(image.cols());
        // apply the optimal cols and rows size to the image
        copyMakeBorder(image, padded, 0, addPixelRows - image.rows(), 0, addPixelCols - image.cols(),
                BORDER_CONSTANT, Scalar.all(0));

        return padded;
    }

    /**
     * Optimize the magnitude of the complex image obtained from the DFT, to
     * improve its visualization
     *
     * @param complexImage the complex image obtained from the DFT
     * @return the optimized image
     */
    private Mat createOptimizedMagnitude(Mat complexImage) {
        // init
        MatVector newPlanes = new MatVector();
        Mat mag = new Mat();
        // split the comples image in two planes
        split(complexImage, newPlanes);
        // compute the magnitude
        magnitude(newPlanes.get(0), newPlanes.get(1), mag);

        // move to a logarithmic scale
        add(Mat.ones(mag.size(), CV_32F).asMat(), mag, mag);
        log(mag, mag);
        // optionally reorder the 4 quadrants of the magnitude image
        this.shiftDFT(mag);
        // normalize the magnitude image for the visualization since both JavaFX
        // and OpenCV need images with value between 0 and 255
        // convert back to CV_8UC1
        mag.convertTo(mag, CV_8UC1);
        normalize(mag, mag, 0, 255, NORM_MINMAX, CV_8UC1, Mat.EMPTY);

        // you can also write on disk the resulting image...
        // Imgcodecs.imwrite("../magnitude.png", mag);

        return mag;
    }

    /**
     * Reorder the 4 quadrants of the image representing the magnitude, after
     * the DFT
     *
     * @param image the {@link Mat} object whose quadrants are to reorder
     */
    private void shiftDFT(Mat image) {
        image = image.apply(new Rect(0, 0, image.cols() & -2, image.rows() & -2));
        int cx = image.cols() / 2;
        int cy = image.rows() / 2;

        Mat q0 = new Mat(image, new Rect(0, 0, cx, cy));
        Mat q1 = new Mat(image, new Rect(cx, 0, cx, cy));
        Mat q2 = new Mat(image, new Rect(0, cy, cx, cy));
        Mat q3 = new Mat(image, new Rect(cx, cy, cx, cy));

        Mat tmp = new Mat();
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);
    }

    /**
     * Set the current stage (needed for the FileChooser modal window)
     *
     * @param stage the stage
     */
    public void setStage(Stage stage) {
        this.stage = stage;
    }

    /**
     * Convert a Mat object (OpenCV) in the corresponding Image for JavaFX
     *
     * @param frame the {@link Mat} representing the current frame
     * @return the {@link Image} to show
     */
    private Image mat2Image(Mat frame) {
        // create a temporary buffer
        byte[] buffer = new byte[frame.cols() * frame.rows() * frame.channels()];
        // encode the frame in the buffer, according to the PNG format
        imencode(".png", frame, buffer);

        // build and return an Image created from the image encoded in the
        // buffer
        return new Image(new ByteArrayInputStream(buffer));
    }
}
