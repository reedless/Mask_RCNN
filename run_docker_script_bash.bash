# Run bash container
docker run --name mask-rcnn -it --rm -p 8888:8888 -p 6006:6006 -v ~/Mask_RCNN/:/host mask_rcnn
