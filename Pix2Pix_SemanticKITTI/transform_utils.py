
from torchvision import transforms

rgb_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0438, 0.0443, 0.0428], std=[0.1523, 0.1522, 0.1471])
])

lidar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1366], std=[0.1506])
])

intensity_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4257], std=[0.2276])
])

incidence_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3771], std=[0.2514])
])

binary_transform = transforms.Compose([
    transforms.ToTensor(),
    
])
#transforms.Normalize(mean=[0.7287], std=[0.4430])
color_transform = transforms.Compose([
    transforms.ToTensor(),
    
])
#transforms.Normalize(mean=[0.1288], std=[0.3349])
label_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3416], std=[0.2252])
])

reflectance_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2683], std=[0.2696])
])



