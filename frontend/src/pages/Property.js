import {
  View,
  Text,
  Image,
  StyleSheet,
  TouchableOpacity,
  ToastAndroid,
  TextInput,
  ScrollView,
} from 'react-native';
import React, {useState} from 'react';
import {Avatar} from '@rneui/base';
import {useNavigation, useRoute} from '@react-navigation/native';
import firestore from '@react-native-firebase/firestore';

const Property = () => {
  const navigation = useNavigation();
  const route = useRoute();
  const {userD, propertyData} = route.params;
  const [price, setPrice] = useState(propertyData.price.toString());
  const [isUpdated, setIsUpdated] = useState(false);

  // console.log(PropertyData)
  const onSubmit = async () => {
    try {
      const user = {};
      await firestore()
        .collection('property') // Create a subcollection for properties
        .add(propertyData);
      ToastAndroid.show('Property added successfully', ToastAndroid.SHORT);
      navigation.navigate('Tabs', {userD});
    } catch (error) {
      console.error(error.message);
    }
  };

  const onRequest = async () => {
    try {
      const data = {
        contracterId: userD.email,
        contracterName: userD.name,
        contracterPrice: price,
        customerId: propertyData.userId,
        customerName: propertyData.userName,
        propertyId: propertyData.propertyId,
        status: 'Requested',
        // Use the updated price
      };

      // Update the property with the new price and status
      await firestore().collection('contract').doc().set(data, {merge: true});
      ToastAndroid.show(
        'Property contract requested successfully',
        ToastAndroid.SHORT,
      );
      navigation.navigate('ContracterPage', {userD});
    } catch (error) {
      console.error(error.message);
    }
  };

  return (
    <View style={{flex: 1,marginBottom:10}}>
      <ScrollView>
        <View style={styles.imageContainer}>
          <Image source={require('../assets/house.jpg')} style={styles.image} />
          <View style={styles.pricetag}>
            <Text style={styles.pricetagText}>₹{propertyData.price}</Text>
          </View>
        </View>
        <View style={styles.location}>
          <Avatar
            source={require('../assets/location.png')}
            containerStyle={{marginTop: 10}}
          />
          <Text style={{fontSize: 18, marginLeft: -10}}>
            {propertyData.location}
          </Text>
        </View>

        <View style={styles.containers}>
          <View style={styles.box}>
            <Image
              source={require('../assets/bedroom.png')}
              style={styles.boxImage}
            />
            <Text style={styles.boxText}>{propertyData.Bedrooms} Bed</Text>
          </View>

          <View style={styles.box}>
            <Image
              source={require('../assets/wifi.png')}
              style={styles.boxImage}
            />
            <Text style={styles.boxText}>{propertyData.Wifi}</Text>
          </View>

          <View style={styles.box}>
            <Image
              source={require('../assets/rooms.png')}
              style={styles.boxImage}
            />
            <Text style={styles.boxText}>7Rooms</Text>
          </View>

          <View style={styles.box}>
            <Image
              source={require('../assets/directions.png')}
              style={styles.boxImage}
            />
            <Text style={styles.boxText}>{propertyData.LotArea} sqft</Text>
          </View>
        </View>
        <Text
          style={{
            padding: 10,
            fontSize: 18,
            fontWeight: '500',
            color: 'black',
            fontSize:20
          }}>
          Details
        </Text>

        <View
          style={styles.rows}>
          <Text style={styles.text}>
          MSZoning: {propertyData.MSZoning}
          </Text>
          <Text style={styles.text}>
          Street: {propertyData.Street}
            </Text>
        </View>

        <View style={styles.rows}>
          <Text style={styles.text}>
          Condition1: {propertyData.Condition1}
          </Text>
          <Text style={styles.text}>
          OverallCond: {propertyData.OverallCond}
            </Text>
        </View>

        <View style={styles.rows}>
          <Text style={styles.text}>
          RoofStyle: {propertyData.RoofStyle}
          </Text>
          <Text style={styles.text}>
          RoofMatl: {propertyData.RoofMatl}
            </Text>
        </View>

        <View style={styles.rows}>
          <Text style={styles.text}>
          Foundation: {propertyData.Foundation}
          </Text>
          <Text style={styles.text}>
          BsmtQual: {propertyData.BsmtQual}
            </Text>
        </View>

        <View style={styles.rows}>
          <Text style={styles.text}>
          Heating: {propertyData.Heating}
          </Text>
          <Text style={styles.text}>
          HeatingQC: {propertyData.HeatingQC}
            </Text>
        </View>

        <View style={styles.rows}>
          <Text style={styles.text}>
          CentralAir: {propertyData.CentralAir}
          </Text>
          <Text style={styles.text}>
          Electrical: {propertyData.Electrical}
            </Text>
        </View>

        <View style={styles.rows}>
          <Text style={styles.text}>
          FullBath: {propertyData.FullBath}
          </Text>
          <Text style={styles.text}>
          KitchenQual: {propertyData.KitchenQual}
            </Text>
        </View>

        <View style={styles.rows}>
          <Text style={styles.text}>
          TotRmsAbvGrd: {propertyData.TotRmsAbvGrd}
          </Text>
          <Text style={styles.text}>
          Functional: {propertyData.Functional}
            </Text>
        </View>

        <View style={styles.rows}>
          <Text style={styles.text}>
          GarageType: {propertyData.GarageType}
          </Text>
          <Text style={styles.text}>
            StaffQuarter: {propertyData.StaffQuarter}
            </Text>
        </View>


        {userD.userType === 'user' ? (
          <TouchableOpacity style={styles.btnsubmit} onPress={onSubmit}>
            <Text style={[styles.pricetagText, {marginTop: 15, fontSize: 20}]}>
              Publish
            </Text>
          </TouchableOpacity>
        ) : (
          <>
            <TextInput
              style={styles.priceInput}
              value={price} // Use the price state variable directly
              onChangeText={text => setPrice(text)}
            />
            <TouchableOpacity style={styles.btnsubmit} onPress={onRequest}>
              <Text
                style={[styles.pricetagText, {marginTop: 15, fontSize: 20}]}>
                Request
              </Text>
            </TouchableOpacity>
          </>
        )}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  imageContainer: {
    position: 'relative',
    marginTop: 10,
  },
  image: {
    width: '95%',
    height: 300,
    borderRadius: 10,
    alignSelf: 'center',
  },
  pricetag: {
    backgroundColor: '#ff8717',
    bottom: 10, // Adjust the position as needed
    width: 100, // Adjust the width as needed
    paddingVertical: 5,
    borderRadius: 20,
    marginTop: -280,
    marginRight: 20,
    alignSelf: 'flex-end',
  },
  pricetagText: {
    color: 'white',
    textAlign: 'center',
  },
  location: {
    marginTop: 250,
    width: '95%',
    height: 50,
    backgroundColor: 'white',
    alignSelf: 'center',
    alignItems: 'center',
    borderBottomLeftRadius: 10,
    borderBottomRightRadius: 10,
    flexDirection: 'row',
  },
  containers: {
    flexDirection: 'row',
    alignItems: 'center',
    marginLeft: 10,
    padding: 10,
    marginTop: 20,
  },
  box: {
    width: 75,
    height: 75,
    backgroundColor: 'white',
    alignItems: 'center',
    borderRadius: 10,
    marginRight: 10,
  },
  boxImage: {
    width: 40,
    height: 40,
    marginTop: 5,
  },
  boxText: {
    fontSize: 15,
    marginTop: 5,
    fontWeight: '300',
    color: 'black',
  },
  btnsubmit: {
    backgroundColor: '#ff8717',
    height: 55,
    width: '95%',
    marginTop: 30,
    borderRadius: 30,
    alignSelf: 'center',
  },
  priceInput: {
    // backgroundColor: '#ff8717',
    textAlign: 'center',
    height: 55,
    width: '95%',
    marginTop: 30,
    borderRadius: 30,
    alignSelf: 'center',
    borderWidth: 1,
    color: 'black',
  },rows:{
    flexDirection: 'row', 
    marginLeft: 10, 
    marginRight: 10
  },
  text:{
    color: 'black',
    width:200,
  fontSize:16}
});

export default Property;
