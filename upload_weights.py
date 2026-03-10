from roboflow import Roboflow

# 1. Paste the API key you copied from the website right here
rf = Roboflow(api_key="d6Weko7t4TxWsLTnBjCK")

# 2. This points to your exact Roboflow workspace and project
project = rf.workspace("stephen-ikolg").project("main_project-jhxvk")
version = project.version(1)

# 3. Upload the custom model brain! 
# (Roboflow automatically looks inside this folder for the /weights/best.pt file)
print("Uploading weights to Roboflow...")
version.deploy(model_type="yolov8", model_path="runs/detect/train3")
print("Upload complete! You can now use Label Assist.")