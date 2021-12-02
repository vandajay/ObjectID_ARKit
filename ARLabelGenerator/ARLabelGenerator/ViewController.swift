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
    
    private var MLModel = try! Resnet50(configuration: MLModelConfiguration())
    private var hitTestResult :ARHitTestResult!
    private var VRH = [VNRequest]()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        sceneView.delegate = self
        sceneView.showsStatistics = true
        let scene = SCNScene()
        sceneView.scene = scene
        registerGestureRecognizers()
    }
    
    private func registerGestureRecognizers() {
        let tapGestureRecognizer = UITapGestureRecognizer(target: self, action: #selector(screenTap))
        self.sceneView.addGestureRecognizer(tapGestureRecognizer)
    }
    @objc func screenTap(recognizer :UIGestureRecognizer) {
        let sceneView = recognizer.view as! ARSCNView
        let tapObject = self.sceneView.center
    
        guard let currentFrame = sceneView.session.currentFrame else { return }
        let hitTestResults = sceneView.hitTest(tapObject, types: .featurePoint)
        if hitTestResults.isEmpty { return }
        guard let hitTestResult = hitTestResults.first else { return }
        self.hitTestResult = hitTestResult
        let pixelBuffer = currentFrame.capturedImage
        performVisionRequest(pixelBuffer: pixelBuffer)
    }
    private func displayPredictions(text :String) {
        let node = createText(text: text)
        node.position = SCNVector3(self.hitTestResult.worldTransform.columns.3.x, self.hitTestResult.worldTransform.columns.3.y, self.hitTestResult.worldTransform.columns.3.z)
        self.sceneView.scene.rootNode.addChildNode(node)
    }
    
    private func createText(text: String) -> SCNNode {
        let parentNode = SCNNode()
        let anchor = SCNSphere(radius: 0.01)
        let anchorMaterial = SCNMaterial()
        anchorMaterial.diffuse.contents = UIColor.blue
        anchor.firstMaterial = anchorMaterial
        let anchorNode = SCNNode(geometry: anchor)
        let ARText = SCNText(string: text, extrusionDepth: 0)
        ARText.alignmentMode = convertFromCATextLayerAlignmentMode(CATextLayerAlignmentMode.center)
        ARText.firstMaterial?.specular.contents = UIColor.white
        ARText.firstMaterial?.diffuse.contents = UIColor.purple
        ARText.firstMaterial?.isDoubleSided = true
        let font = UIFont(name: "Noteworthy", size: 0.15)
        ARText.font = font
        let textNode = SCNNode(geometry: ARText)
        textNode.scale = SCNVector3Make(0.2, 0.2, 0.2)
        parentNode.addChildNode(anchorNode)
        parentNode.addChildNode(textNode)
        return parentNode
    }
    
    private func performVisionRequest(pixelBuffer :CVPixelBuffer) {
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
        visionRequest.imageCropAndScaleOption = .centerCrop
        self.VRH = [visionRequest]
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .upMirrored, options: [:])
        DispatchQueue.global().async {
            try! imageRequestHandler.perform(self.VRH)
        }
        
    }
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        let configuration = ARWorldTrackingConfiguration()
        sceneView.session.run(configuration)
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        sceneView.session.pause()
    }

}

// Helper function
fileprivate func convertFromCATextLayerAlignmentMode(_ input: CATextLayerAlignmentMode) -> String {
    return input.rawValue
}
