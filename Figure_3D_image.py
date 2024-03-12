import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


def plot_3d_multi(image1, image2=None, image3=None, image4=None,
                  image5=None,  image6=None, threshold=-300, elev=0, azim=90):

    face_color1 = 'red'
    face_color2 = 'green'
    face_color3 = 'blue'
    # face_color1 = 'sandybrown'
    # face_color2 = 'cyan'
    # face_color3 = 'gold'
    face_color4 = 'sandybrown'
    face_color5 = 'cyan'
    face_color6 = 'gold'
    ####
    # Anterior view: image1.transpose(1,0,2); superior view: image1.transpose(0,1,2); left view: image1.transpose(2,1,0)
    ####
    p = image1.transpose(1,0,2)
    verts1, faces1, normals1, values1 = measure.marching_cubes(p, threshold)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    mesh1 = Poly3DCollection(verts1[faces1], alpha=0.3)
    mesh1.set_facecolor(face_color1)
    ax.add_collection3d(mesh1)

    if not image2 is None:
        p2 = image2.transpose(1,0,2)
        verts2, faces2, normals2, values2 = measure.marching_cubes(p2, threshold)
        mesh2 = Poly3DCollection(verts2[faces2], alpha=0.5)
        mesh2.set_facecolor(face_color2)
        ax.add_collection3d(mesh2)


    if not image3 is None:
        p3 = image3.transpose(1,0,2)
        verts3, faces3, normals3, values3 = measure.marching_cubes(p3, threshold)
        mesh3 = Poly3DCollection(verts3[faces3], alpha=0.8)
        mesh3.set_facecolor(face_color3)
        ax.add_collection3d(mesh3)


    if not image4 is None:
        p4 = image4.transpose(1,0,2)
        verts4, faces4, normals4, values4 = measure.marching_cubes(p4, threshold)
        mesh4 = Poly3DCollection(verts4[faces4], alpha=0.8)
        mesh4.set_facecolor(face_color4)
        ax.add_collection3d(mesh4)

    if not image5 is None:
        p5 = image5.transpose(1,0,2)
        verts5, faces5, normals5, values5 = measure.marching_cubes(p5, threshold)
        mesh5 = Poly3DCollection(verts5[faces5], alpha=0.8)
        mesh5.set_facecolor(face_color5)
        ax.add_collection3d(mesh5)

    if not image6 is None:
        p6 = image6.transpose(1,0,2)
        verts6, faces6, normals6, values6 = measure.marching_cubes(p6, threshold)
        mesh6 = Poly3DCollection(verts6[faces6], alpha=0.8)
        mesh6.set_facecolor(face_color6)
        ax.add_collection3d(mesh6)
    
    # ax.axis('off')
    ax.set_xlim(0, round(p.shape[0]*1))
    ax.set_ylim(0, round(p.shape[1]*1))
    ax.set_zlim(0, round(p.shape[2]*1))

    # ax.set_facecolor([2/256, 37/256, 112/256])
    plt.title(f'elev={elev}, azim={azim}')
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()


def main():
    # Paths to structure images
    path_struct1 = './code_test_folder/3d_slicer_checker/pred/s0009.nii.gz'
    path_struct2 = './code_test_folder/3d_slicer_checker/gt/s0009.nii.gz'
    path_struct3 = ''
    path_struct4 = ''
    struct1_image = sitk.ReadImage(path_struct1)
    struct1 = sitk.GetArrayViewFromImage(struct1_image)

    struct2_image = sitk.ReadImage(path_struct2)
    struct2 = sitk.GetArrayViewFromImage(struct2_image)

    sub = struct1 - struct2

    # struct3_image = sitk.ReadImage(path_struct3)
    # struct3 = sitk.GetArrayViewFromImage(struct3_image)
    #
    # struct4_image = sitk.ReadImage(path_struct4)
    # struct4 = sitk.GetArrayViewFromImage(struct4_image)
    plot_3d_multi(sub, threshold=0.5)

    # plot_3d_multi(struct1, struct2, struct3, struct4, threshold=0.5)

if __name__ == '__main__':
    main()
