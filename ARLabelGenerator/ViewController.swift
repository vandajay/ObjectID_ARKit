//
//  ViewController.swift
//  ARLabelGenerator
//
//  Created by Brenden York on 12/2/21.
//

import UIKit
import SceneKit
import ARKit
import Vision

class ViewController: UIViewController, ARSCNViewDelegate {

    @IBOutlet var sceneView: ARSCNView!
    
    /******************************************************************
     Download and add Resnet50 model to project.
     Setup configuration and vision request handler.
     ******************************************************************/

    private var MLModel = try! Resnet50(configuration: MLModelConfiguration())
    private var hitTestResult :ARHitTestResult!
    private var VRH = [VNRequest]()
    
    /*******************************************
     Load the scene, show statistics for monitoring application
    ***********************************************************/
    override func viewDidLoad() {
        super.viewDidLoad()
        sceneView.delegate = self
        sceneView.showsStatistics = false
        
        sceneView.autoenablesDefaultLighting = true

        let scene = SCNScene()
        
        sceneView.scene = scene
        
        // Function for setting labels via user screen taps
        registerGestureRecognizers()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        // Create a session configuration
        let configuration = ARWorldTrackingConfiguration()
        
        // Enable plane detection
        configuration.planeDetection = .horizontal
        
        // Run the view's session
        sceneView.session.run(configuration)
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        sceneView.session.pause()
    }
    
    /*
     Hide status bar
     */
    override var prefersStatusBarHidden : Bool {
        return true
    }
    
    /*******************************************
     Register a Gesture recognizer. Picks up tapping on screen anywhere.
    ***********************************************************/
    
    private func registerGestureRecognizers() {
        let tapGestureRecognizer = UITapGestureRecognizer(target: self, action: #selector(screenTap))
        self.sceneView.addGestureRecognizer(tapGestureRecognizer)
    }
    /*******************************************
     screenTap Will be called whenever user taps on screen.
     Gets touch location, and centers it on object.
    ***********************************************************/
    @objc func screenTap(recognizer :UIGestureRecognizer) {
        let sceneView = recognizer.view as! ARSCNView
        let tapObject = self.sceneView.center
        
        // Set center of screen to place label
        let screenCentre : CGPoint = CGPoint(x: self.sceneView.bounds.midX, y: self.sceneView.bounds.midY)

        /*******************************************
        gets current Frame from session (whats in view of camera)
        hitTest Result acts somewhat like a plane, We get the touch location
        Then check for feature points and return if empty if not hitTestResults is set to itself.
        ***********************************************************/
        guard let currentFrame = sceneView.session.currentFrame else { return }
        let hitTestResults = sceneView.hitTest(screenCentre, types: .featurePoint)
        if hitTestResults.isEmpty { return }
        guard let hitTestResult = hitTestResults.first else { return }
        self.hitTestResult = hitTestResult
        /*******************************************
         A Core Video pixel buffer is an image buffer that holds pixels in main memory.
         Pixel buffer is required for the vision request.
        ***********************************************************/
        let pixelBuffer = currentFrame.capturedImage
        performVisionRequest(pixelBuffer: pixelBuffer)
    }
    
    /******************************************************************
     Generates a node represeting the point at which a label is referencing pass in text from prediction
     hitTestResults contains everything related to position of where the user clicked. Uses cordinates
     of where user clicked to place sphere and aid  in text placemnet above sphere anchor.
    *******************************************************************/
    private func displayPredictions(text :String) {
        let node = createText(text: text)
        node.position = SCNVector3(self.hitTestResult.worldTransform.columns.3.x, self.hitTestResult.worldTransform.columns.3.y, self.hitTestResult.worldTransform.columns.3.z)
        self.sceneView.scene.rootNode.addChildNode(node)
    }
    
    /*******************************************
     akes in text and returns you a node.
     creates anchor point for a sphere that is created an appears where the user clicks.
     text appears above the achor sphere.
     Sphere and text color added through firstMatireal.diffuse
     scale and size of text, and anchor added for more practical appreance
    ***********************************************************/
    private func createText(text: String) -> SCNNode {
        
        let labelDepth : Float = 0.1

        
        let parentNode = SCNNode()
        let anchor = SCNCone(topRadius: 0.02, bottomRadius: 0, height: 0.05)
        let anchorMaterial = SCNMaterial()
        anchorMaterial.diffuse.contents = UIColor.green
        anchorMaterial.specular.contents = UIColor.white

        anchor.firstMaterial = anchorMaterial
        let anchorNode = SCNNode(geometry: anchor)
        let ARText = SCNText(string: text, extrusionDepth: 0.08)

//        ARText.alignmentMode = convertFromCATextLayerAlignmentMode(CATextLayerAlignmentMode.center)
        ARText.firstMaterial?.specular.contents = UIColor.white
        ARText.firstMaterial?.diffuse.contents = UIColor.green
        ARText.firstMaterial?.isDoubleSided = true
        ARText.chamferRadius = CGFloat(labelDepth)
        
        

        let font = UIFont(name: "Noteworthy", size: 0.2)
        ARText.font = font
        let textNode = SCNNode(geometry: ARText)
        textNode.scale = SCNVector3Make(0.2, 0.2, 0.2)
        
        parentNode.addChildNode(anchorNode)
        parentNode.addChildNode(textNode)
        
        // Constrain parentNode (Label) to always face you
        let billboardConstraint = SCNBillboardConstraint()
        billboardConstraint.freeAxes = SCNBillboardAxis.Y
        parentNode.constraints = [billboardConstraint]
        
        return parentNode
    }
    
    /*******************************************
     Create a vision model and request pass through the model in which the request is going to run.
          handle errors with a return if nil. Creates observation of what CoreML believes the item is, with
          a confidence rating.
    ***********************************************************/
    private func performVisionRequest(pixelBuffer: CVPixelBuffer) {
        let visionRequestModel = try! VNCoreMLModel(for: self.MLModel.model)
        let visionRequest = VNCoreMLRequest(model: visionRequestModel) { visionRequest, error in
            if error != nil { return }
            guard let observations = visionRequest.results else { return }
            let observation = observations.first as! VNClassificationObservation
            print("Name \(observation.identifier) and confidence is \(observation.confidence)")
            DispatchQueue.main.async {
                self.displayPredictions(text: observation.identifier)
            }
        }
        
        /*******************************************
        Takes the center of the current frame, crops the image to contain the touched object feeds
        cropped image to coreML  and attempts to identify the item. Populates vision reuqest array.
        This is an array of vision reuqest. (can have multiple)
        ***********************************************************/
        visionRequest.imageCropAndScaleOption = .centerCrop
        self.VRH = [visionRequest]
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .upMirrored, options: [:])
        DispatchQueue.global().async {
            try! imageRequestHandler.perform(self.VRH)
        }
    }
}

//// Helper function for text alignmnet
//fileprivate func convertFromCATextLayerAlignmentMode(_ input: CATextLayerAlignmentMode) -> String {
//    return input.rawValue
//}

extension UIFont {
    // Based on: https://stackoverflow.com/questions/4713236/how-do-i-set-bold-and-italic-on-uilabel-of-iphone-ipad
    func withTraits(traits:UIFontDescriptor.SymbolicTraits...) -> UIFont {
        let descriptor = self.fontDescriptor.withSymbolicTraits(UIFontDescriptor.SymbolicTraits(traits))
        return UIFont(descriptor: descriptor!, size: 0)
    }
}
