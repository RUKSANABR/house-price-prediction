import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
} from 'react-native';
import React, {useState} from 'react';
import {Slider, Icon} from '@rneui/themed';
import {useNavigation} from '@react-navigation/native';

const Home = () => {
  const navigation = useNavigation();

  const [type, setType] = useState();
  const [furniture, setFurniture] = useState();
  const [bedrooms, setBedrooms] = useState();
  const [bathrooms, setBathrooms] = useState();
  const [propertySize, setPropertySize] = useState();

  const color = () => {
    // let r = interpolate(255, 0);
    // let g = interpolate(0, 255);
    // let b = interpolate(0, 0);
    // return `rgb(${r},${g},${b})`;
  };

  return (
    <View style={styles.container}>
      <ScrollView>
        {/*  */}
        <Text style={styles.head}>Home</Text>

        <Text style={styles.formText}>Property Type</Text>
        <View style={styles.btnGroup}>
          <TouchableOpacity
            style={[
              styles.btn,
              type === 'House' ? {backgroundColor: '#ff8717'} : null,
            ]}
            onPress={() => setType('House')}>
            <Text
              style={[
                styles.btnText,
                type === 'House' ? {color: 'white'} : null,
              ]}>
              House
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.btn,
              type === 'Apartment' ? {backgroundColor: '#ff8717'} : null,
            ]}
            onPress={() => setType('Apartment')}>
            <Text
              style={[
                styles.btnText,
                type === 'Apartment' ? {color: 'white'} : null,
              ]}>
              Apartment
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.btn,
              type === 'Flat' ? {backgroundColor: '#ff8717'} : null,
            ]}
            onPress={() => setType('Flat')}>
            <Text
              style={[
                styles.btnText,
                type === 'Flat' ? {color: 'white'} : null,
              ]}>
              Flat
            </Text>
          </TouchableOpacity>
        </View>
        <Text style={styles.formText}>Bedrooms</Text>
        {/*  */}
        <View style={styles.btnGroup}>
          <TouchableOpacity
            style={[
              styles.btn,
              bedrooms === 1 ? {backgroundColor: '#ff8717'} : null,
            ]}
            onPress={() => setBedrooms(1)}>
            <Text
              style={[
                styles.btnText,
                bedrooms === 1 ? {color: 'white'} : null,
              ]}>
              1
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.btn,
              bedrooms === 2 ? {backgroundColor: '#ff8717'} : null,
            ]}
            onPress={() => setBedrooms(2)}>
            <Text
              style={[
                styles.btnText,
                bedrooms === 2 ? {color: 'white'} : null,
              ]}>
              2
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.btn,
              bedrooms === 3 ? {backgroundColor: '#ff8717'} : null,
            ]}
            onPress={() => setBedrooms(3)}>
            <Text
              style={[
                styles.btnText,
                bedrooms === 3 ? {color: 'white'} : null,
              ]}>
              3
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.btn,
              bedrooms === 4 ? {backgroundColor: '#ff8717'} : null,
            ]}
            onPress={() => setBedrooms(4)}>
            <Text
              style={[
                styles.btnText,
                bedrooms === 4 ? {color: 'white'} : null,
              ]}>
              4
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[
              styles.btn,
              bedrooms === 5 ? {backgroundColor: '#ff8717'} : null,
            ]}
            onPress={() => setBedrooms(5)}>
            <Text
              style={[
                styles.btnText,
                bedrooms === 5 ? {color: 'white'} : null,
              ]}>
              5
            </Text>
          </TouchableOpacity>
        </View>
        {/*  */}
        <Text style={styles.formText}>Bathrooms</Text>
        {/*  */}
        <View style={styles.btnGroup}>
          <TouchableOpacity
            style={[
              styles.btn,
              bathrooms === 1 ? {backgroundColor: '#ff8717'} : null,
            ]}
            onPress={() => setBathrooms(1)}>
            <Text
              style={[
                styles.btnText,
                bathrooms === 1 ? {color: 'white'} : null,
              ]}>
              1
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.btn,
              bathrooms === 2 ? {backgroundColor: '#ff8717'} : null,
            ]}
            onPress={() => setBathrooms(2)}>
            <Text
              style={[
                styles.btnText,
                bathrooms === 2 ? {color: 'white'} : null,
              ]}>
              2
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.btn,
              bathrooms === 3 ? {backgroundColor: '#ff8717'} : null,
            ]}
            onPress={() => setBathrooms(3)}>
            <Text
              style={[
                styles.btnText,
                bathrooms === 3 ? {color: 'white'} : null,
              ]}>
              3
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.btn,
              bathrooms === 4 ? {backgroundColor: '#ff8717'} : null,
            ]}
            onPress={() => setBathrooms(4)}>
            <Text
              style={[
                styles.btnText,
                bathrooms === 4 ? {color: 'white'} : null,
              ]}>
              4
            </Text>
          </TouchableOpacity>
        </View>
        {/*  */}

        <Text style={styles.formText}>Furniture</Text>
        <View style={styles.btnGroup}>
          <TouchableOpacity
            style={[
              styles.btn,
              furniture === 'Fully-Furnished'
                ? {backgroundColor: '#ff8717'}
                : null,
            ]}
            onPress={() => setFurniture('Fully-Furnished')}>
            <Text
              style={[
                styles.btnText,
                furniture === 'Fully-Furnished' ? {color: 'white'} : null,
              ]}>
              Fully-Furnished
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.btn,
              furniture === 'Semi-Furnished'
                ? {backgroundColor: '#ff8717'}
                : null,
            ]}
            onPress={() => setFurniture('Semi-Furnished')}>
            <Text
              style={[
                styles.btnText,
                furniture === 'Semi-Furnished' ? {color: 'white'} : null,
              ]}>
              Semi-Furnished
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.btn,
              furniture === 'No-furnished'
                ? {backgroundColor: '#ff8717'}
                : null,
            ]}
            onPress={() => setFurniture('No-furnished')}>
            <Text
              style={[
                styles.btnText,
                furniture === 'No-furnished' ? {color: 'white'} : null,
              ]}>
              No-furnished
            </Text>
          </TouchableOpacity>
        </View>
        {/*  */}

        <Text style={styles.formText}>Property Size</Text>
        <View style={[styles.contentView]}>
          <Text
            style={[styles.formText, {color: '#1ebf94', fontWeight: '500'}]}>
            Up to {propertySize}sqft
          </Text>

          <Slider
            value={propertySize}
            onValueChange={setPropertySize}
            maximumValue={6000}
            minimumValue={1000}
            step={1}
            allowTouchTrack
            trackStyle={{height: 5, backgroundColor: '#1ebf94'}}
            thumbStyle={{
              height: 20,
              width: 10,
              backgroundColor: '#1ebf94',
            }}
            thumbProps={{
              children: (
                <Icon
                  name="circle"
                  type="font-awesome"
                  size={10}
                  reverse
                  containerStyle={{bottom: 10, right: 10}}
                  color={'#1ebf94'}
                />
              ),
            }}
          />
        </View>
        {/*  */}

        <TouchableOpacity
          style={styles.btnsubmit}
          onPress={() => navigation.navigate('Property')}>
          <Text style={[styles.btnText, {fontSize: 18, color: 'white'}]}>
            Predict Price
          </Text>
        </TouchableOpacity>
        {/*  */}
      </ScrollView>
    </View>
  );
};
const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 10,
  },

  head: {
    fontWeight: '500',
    fontSize: 28,
    color: 'black',
    textAlign: 'center',
  },
  formText: {
    fontWeight: '500',
    fontSize: 20,
    marginBottom: 10,
    marginTop: 20,
    color: 'black',
  },
  btnGroup: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  btn: {
    flex: 1,
    borderWidth: 0.25,
    borderColor: '#6B7280',
    borderRadius: 5,
    marginRight: 10,
  },
  btnText: {
    textAlign: 'center',
    paddingVertical: 16,
    fontSize: 15,
    color: 'black',
  },
  contentView: {
    width: '97%',
    height: 100,
    justifyContent: 'center',
    alignItems: 'stretch',
    backgroundColor: 'white',
    padding: 10,
    borderRadius: 10,
  },
  btnsubmit: {
    backgroundColor: '#ff8717',
    height: 55,
    width: '95%',
    marginTop: 30,
    borderRadius: 30,
    alignSelf: 'center',
  },
});
export default Home;
